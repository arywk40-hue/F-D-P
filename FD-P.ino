/**
 * Line Following Robot — ESP32 + L298N
 * ─────────────────────────────────────────────────────────────
 * Features:
 *   • Sensor calibration (manual sweep at boot)
 *   • Relay-feedback auto-tune (Åström–Hägglund) → Ziegler–Nichols PID
 *   • 1-D Kalman filter on the error signal
 *   • PID controller with derivative smoothing + anti-windup
 *   • Optional ε-greedy Q-learning correction layer
 *
 * Bug-fixes vs. original:
 *   1. Forward declarations added (autoTunePID used helpers before definition).
 *   2. peakMax/peakMin reset to current `err` after each zero-crossing, and
 *      per-cycle amplitude is pushed into a ring buffer — averaged over the last
 *      3 stable cycles — so Ku is not sensitive to a single noisy half-cycle.
 *   3. driverEnable(true) moved outside the tune loop (was called every 5 ms).
 *   4. Proper deferred TD update: (prevState, prevAction, prevReward) are carried
 *      across loop iterations so Q is updated with a genuinely new state.
 *   5. randomSeed(esp_random()) removed — Arduino random() is never called;
 *      esp_random() is used directly and needs no seeding.
 *   6. Indentation / brace alignment normalised throughout.
 *   7. safeMap clamp uses long arithmetic to avoid 16-bit overflow on AVR-like
 *      targets (harmless but correct on ESP32 too).
 * ─────────────────────────────────────────────────────────────
 */

#include <Arduino.h>
#include <esp_system.h>   // esp_random()

// ═══════════════════════════════════════════════════════════════
//  PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════
#define PWMA  2
#define AIN1  4
#define AIN2  16

#define PWMB  19
#define BIN1  18
#define BIN2  5

// Enable pin wired to L298N ENA/ENB or module EN.
// Set -1 if not used (bridge is always enabled by a jumper).
static constexpr int STBY = 17;

#define LEFT_SENSOR   25
#define CENTER_SENSOR 26
#define RIGHT_SENSOR  27

// ═══════════════════════════════════════════════════════════════
//  ESP32 LEDC (replaces analogWrite)
// ═══════════════════════════════════════════════════════════════
static constexpr int PWM_FREQ_HZ  = 20000; // 20 kHz — inaudible
static constexpr int PWM_RES_BITS = 8;     // 0 … 255
static constexpr int PWM_CH_A     = 0;
static constexpr int PWM_CH_B     = 1;

// ═══════════════════════════════════════════════════════════════
//  TUNING KNOBS
// ═══════════════════════════════════════════════════════════════
static int   baseSpeed = 180;   // nominal motor PWM (0–255)

// PID — overwritten by autoTunePID() when AUTO_TUNE_PID = true
static float Kp = 40.0f;
static float Ki =  0.0f;
static float Kd = 25.0f;

// Auto-tune switch
static constexpr bool AUTO_TUNE_PID = true;

// Normalised sensor threshold for "line detected" (scale 0..1000).
// Used identically in auto-tune and in the main loop.
static constexpr int DETECT_THRESH = 300;

// Relay-feedback parameters (used only when AUTO_TUNE_PID = true)
static constexpr int     TUNE_BASE_SPEED      = 140;
static constexpr float   TUNE_RELAY_AMPLITUDE = 60.0f;
static constexpr float   TUNE_TARGET_ERROR    = 0.0f;
static constexpr float   TUNE_DEADBAND        = 0.06f;
static constexpr uint32_t TUNE_TIMEOUT_MS     = 7000;
static constexpr uint32_t TUNE_MIN_OSC_MS     = 1200; // reject noise spikes

// RL (ε-greedy Q-learning) parameters
static constexpr float RL_ALPHA   = 0.10f;  // learning rate
static constexpr float RL_GAMMA   = 0.90f;  // discount factor
static constexpr int   RL_EPSILON = 5;      // % exploration

// Kalman noise params
static constexpr float KALMAN_Q = 0.10f;    // process noise
static constexpr float KALMAN_R = 0.80f;    // measurement noise

// ═══════════════════════════════════════════════════════════════
//  STATE VARIABLES
// ═══════════════════════════════════════════════════════════════

// PID
static float pidPrevError      = 0.0f;
static float pidPrevDerivative = 0.0f;
static float pidIntegral       = 0.0f;

// Kalman
static float kalmX = 0.0f;  // state estimate
static float kalmP = 1.0f;  // estimate covariance

// Sensor calibration
static int minL = 4095, maxL = 0;
static int minC = 4095, maxC = 0;
static int minR = 4095, maxR = 0;

// RL
// States 0–4 are positional (on-line); state 5 is the explicit LOST state.
// A dedicated LOST row lets the agent learn a distinct recovery policy rather
// than conflating line-loss with the last known positional state.
static constexpr int RL_STATE_LOST = 5;
static constexpr int RL_NUM_STATES = 6;   // 0..4 positional + 5 lost

static float       Qtable[RL_NUM_STATES][5] = {};
static const float actions[5] = { -30.0f, -10.0f, 0.0f, 10.0f, 30.0f };

// Deferred TD update: carry (s, a, r) from the previous iteration so the
// Q update uses a genuinely new state, not the state measured before the
// motor command had time to move the robot.
struct RLTransition {
    int   state  = -1;   // -1 = no pending update
    int   action =  0;
    float reward =  0.0f;
};
static RLTransition rlPrev;

// ═══════════════════════════════════════════════════════════════
//  FORWARD DECLARATIONS  (required because autoTunePID is defined
//  before the helpers it calls)
// ═══════════════════════════════════════════════════════════════
static int   safeMap(int x, int in_min, int in_max, int out_min, int out_max);
void         readSensors(int &L, int &C, int &R);
float        getError(int L, int C, int R);
float        kalmanFilter(float measurement);
void         setMotor(int A, int B);
static void  stopMotors();
static void  driverEnable(bool on);
static bool  autoTunePID();
float        PID(float error);
int          getState(float error, bool detected);
int          chooseAction(int state);
float        getReward(float error, bool detected);
void         updateQ(int s, int a, float reward, int ns);

// ═══════════════════════════════════════════════════════════════
//  DRIVER HELPERS
// ═══════════════════════════════════════════════════════════════
static void driverEnable(bool on) {
    if (STBY >= 0) {
        digitalWrite(STBY, on ? HIGH : LOW);
    }
}

static void stopMotors() {
    ledcWrite(PWM_CH_A, 0);
    ledcWrite(PWM_CH_B, 0);
    driverEnable(false);
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
}

// ═══════════════════════════════════════════════════════════════
//  AUTO-TUNE  (Relay feedback → Ziegler–Nichols)
//  FIX: peakMax/peakMin now reset to `err` after a zero-crossing
//       so the amplitude is measured per-cycle, not globally.
//  FIX: driverEnable() called once before the loop, not every tick.
// ═══════════════════════════════════════════════════════════════
static bool autoTunePID() {
    // Reset controller & Kalman state so tuning is uncontaminated.
    pidPrevError      = 0.0f;
    pidPrevDerivative = 0.0f;
    pidIntegral       = 0.0f;
    kalmX             = 0.0f;
    kalmP             = 1.0f;

    const uint32_t start = millis();

    // ── Per-cycle ring buffers (index 0 = oldest, N_CYCLES-1 = newest) ──
    // We collect amplitude and period per full oscillation cycle, then
    // average the last N_CYCLES to get a stable Ku / Pu estimate.
    static constexpr int N_CYCLES = 3;
    float    ampBuf[N_CYCLES]    = {};   // half-peak-to-peak per cycle
    uint32_t periodBuf[N_CYCLES] = {};   // full period in ms per cycle
    int      cycleIdx            = 0;    // next slot to write

    float    peakMax  = -1e9f;
    float    peakMin  =  1e9f;
    int      cyclesRecorded = 0;
    uint32_t lastCross = 0;
    bool     driveRight = true;
    float    lastErr    = 0.0f;

    driverEnable(true);  // enable once, outside the loop

    while (millis() - start < TUNE_TIMEOUT_MS) {
        int L, C, R;
        readSensors(L, C, R);

        if (L < DETECT_THRESH && C < DETECT_THRESH && R < DETECT_THRESH) {
            stopMotors();
            Serial.println("AUTO-TUNE: line lost, aborting.");
            return false;
        }

        float err = kalmanFilter(getError(L, C, R));

        // Relay: bang-bang around target with deadband.
        if      (err >  (TUNE_TARGET_ERROR + TUNE_DEADBAND)) driveRight = false;
        else if (err <  (TUNE_TARGET_ERROR - TUNE_DEADBAND)) driveRight = true;

        const float u = driveRight ? TUNE_RELAY_AMPLITUDE : -TUNE_RELAY_AMPLITUDE;
        setMotor(TUNE_BASE_SPEED + (int)u,
                 TUNE_BASE_SPEED - (int)u);

        // Accumulate peaks for the current half-cycle window.
        if (err > peakMax) peakMax = err;
        if (err < peakMin) peakMin = err;

        // Zero-crossing → end of a half-period.
        const bool crossed = (err >= 0.0f && lastErr < 0.0f) ||
                             (err <  0.0f && lastErr >= 0.0f);
        if (crossed) {
            const uint32_t now = millis();
            if (lastCross != 0) {
                const uint32_t halfPeriod = now - lastCross;
                const uint32_t fullPeriod = halfPeriod * 2;

                if (fullPeriod >= TUNE_MIN_OSC_MS) {
                    // Record this cycle's amplitude and period.
                    const int slot       = cycleIdx % N_CYCLES;
                    ampBuf[slot]         = (peakMax - peakMin) * 0.5f;
                    periodBuf[slot]      = fullPeriod;
                    cycleIdx++;
                    cyclesRecorded++;
                }
            }
            lastCross = now;

            // Reset window trackers to current value so the next
            // half-cycle starts from the actual crossing point.
            peakMax = err;
            peakMin = err;

            if (cyclesRecorded >= N_CYCLES + 1) break;  // ring fully populated
        }

        lastErr = err;
        delay(5);
    }

    stopMotors();

    if (cyclesRecorded < N_CYCLES) {
        Serial.printf("AUTO-TUNE: only %d cycles collected (need %d).\n",
                      cyclesRecorded, N_CYCLES);
        return false;
    }

    // ── Average amplitude and period — only over populated slots ────────
    // Slots that were never written hold their zero-initialised values.
    // Including them would silently dilute a the average if cyclesRecorded
    // happened to exactly equal N_CYCLES (ring just filled, no extras).
    // We sum only entries where periodBuf[i] > 0 and recount to be safe.
    float    aSum = 0.0f;
    uint32_t pSum = 0;
    int      validCount = 0;
    for (int i = 0; i < N_CYCLES; ++i) {
        if (periodBuf[i] > 0) {
            aSum += ampBuf[i];
            pSum += periodBuf[i];
            ++validCount;
        }
    }
    if (validCount == 0) {
        Serial.println("AUTO-TUNE: no valid cycles in buffer.");
        return false;
    }
    const float a  = aSum / (float)validCount;
    const float Pu = (float)(pSum / (uint32_t)validCount) / 1000.0f; // seconds

    if (a < 0.02f) {
        Serial.println("AUTO-TUNE: averaged amplitude too small.");
        return false;
    }
    if (Pu <= 0.05f) {
        Serial.println("AUTO-TUNE: averaged period too short (noise?).");
        return false;
    }

    // ── Relay → ultimate gain  Ku = 4d / (π · a) ─────────────────────────
    const float d  = TUNE_RELAY_AMPLITUDE;
    const float Ku = (4.0f * d) / (3.14159265f * a);

    // ── Ziegler–Nichols PID rules ─────────────────────────────────────────
    Kp = 0.60f * Ku;
    Ki = (1.20f * Ku) / Pu;
    Kd = 0.075f * Ku * Pu;

    Serial.printf("AUTO-TUNE OK: a=%.3f (avg %d cycles)  Ku=%.2f  Pu=%.3fs"
                  "  =>  Kp=%.2f  Ki=%.2f  Kd=%.2f\n",
                  a, validCount, Ku, Pu, Kp, Ki, Kd);
    return true;
}

// ═══════════════════════════════════════════════════════════════
//  SENSOR UTILITIES
// ═══════════════════════════════════════════════════════════════
static int safeMap(int x, int in_min, int in_max, int out_min, int out_max) {
    if (in_max <= in_min) return (out_min + out_max) / 2;
    const long v = (long)(x - in_min) * (out_max - out_min)
                 / (long)(in_max - in_min) + out_min;
    return (int)constrain(v, (long)out_min, (long)out_max);
}

void readSensors(int &L, int &C, int &R) {
    L = safeMap(analogRead(LEFT_SENSOR),   minL, maxL, 0, 1000);
    C = safeMap(analogRead(CENTER_SENSOR), minC, maxC, 0, 1000);
    R = safeMap(analogRead(RIGHT_SENSOR),  minR, maxR, 0, 1000);
}

// Weighted centroid: –1 = far left, 0 = centre, +1 = far right
float getError(int L, int C, int R) {
    const long sum = (long)L + (long)C + (long)R;
    if (sum < 50) return pidPrevError; // line lost → hold last error

    const float e = (-1.0f * L + 0.0f * C + 1.0f * R) / (float)sum;
    return constrain(e, -1.0f, 1.0f);
}

// ═══════════════════════════════════════════════════════════════
//  KALMAN FILTER  (scalar, 1-D)
// ═══════════════════════════════════════════════════════════════
float kalmanFilter(float measurement) {
    kalmP += KALMAN_Q;
    const float K = kalmP / (kalmP + KALMAN_R);
    kalmX += K * (measurement - kalmX);
    kalmP *= (1.0f - K);
    return kalmX;
}

// ═══════════════════════════════════════════════════════════════
//  PID CONTROLLER
// ═══════════════════════════════════════════════════════════════
float PID(float error) {
    // Dead-zone: suppress very small errors (sensor noise)
    if (fabsf(error) < 0.05f) error = 0.0f;

    // Conditional integration (only when error is small → prevents windup)
    if (fabsf(error) < 0.5f) pidIntegral += error;
    pidIntegral = constrain(pidIntegral, -300.0f, 300.0f);

    // Low-pass on derivative to reduce noise amplification
    const float derivative = 0.7f * (error - pidPrevError)
                           + 0.3f * pidPrevDerivative;

    const float output = Kp * error + Ki * pidIntegral + Kd * derivative;

    pidPrevError      = error;
    pidPrevDerivative = derivative;
    return output;
}

// ═══════════════════════════════════════════════════════════════
//  MOTOR DRIVER
// ═══════════════════════════════════════════════════════════════
void setMotor(int A, int B) {
    A = constrain(A, -255, 255);
    B = constrain(B, -255, 255);

    digitalWrite(AIN1, A >= 0 ? HIGH : LOW);
    digitalWrite(AIN2, A <  0 ? HIGH : LOW);
    ledcWrite(PWM_CH_A, (uint32_t)abs(A));

    digitalWrite(BIN1, B >= 0 ? HIGH : LOW);
    digitalWrite(BIN2, B <  0 ? HIGH : LOW);
    ledcWrite(PWM_CH_B, (uint32_t)abs(B));
}

// ═══════════════════════════════════════════════════════════════
//  Q-LEARNING  (ε-greedy, tabular)
// ═══════════════════════════════════════════════════════════════
// detected=false → RL_STATE_LOST (5), regardless of the stale error value.
// This prevents the agent from attributing a line-loss penalty to whatever
// positional state happened to be active when the line disappeared.
int getState(float error, bool detected) {
    if (!detected) return RL_STATE_LOST;
    if (error < -0.50f) return 0;   // hard left
    if (error < -0.05f) return 1;   // slight left
    if (error <=  0.05f) return 2;  // on-line
    if (error <   0.50f) return 3;  // slight right
    return 4;                        // hard right
}

int chooseAction(int state) {
    if ((int)(esp_random() % 100) < RL_EPSILON)
        return (int)(esp_random() % 5);     // explore

    int best = 0;
    for (int i = 1; i < 5; ++i) {
        // Qtable row `state` is always valid: state ∈ [0, RL_NUM_STATES).
        if (Qtable[state][i] > Qtable[state][best]) best = i;
    }
    return best;
}

float getReward(float error, bool detected) {
    if (!detected) return -50.0f;
    return 10.0f - fabsf(error) * 20.0f;
}

void updateQ(int s, int a, float reward, int ns) {
    float maxQ = Qtable[ns][0];
    for (int i = 1; i < 5; ++i) {
        if (Qtable[ns][i] > maxQ) maxQ = Qtable[ns][i];
    }
    Qtable[s][a] += RL_ALPHA * (reward + RL_GAMMA * maxQ - Qtable[s][a]);
}

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
    // ── GPIO ──────────────────────────────────────────────────
    for (int pin : {AIN1, AIN2, BIN1, BIN2})
        pinMode(pin, OUTPUT);
    if (STBY >= 0) pinMode(STBY, OUTPUT);

    for (int pin : {LEFT_SENSOR, CENTER_SENSOR, RIGHT_SENSOR})
        pinMode(pin, INPUT);

    // ── LEDC PWM ──────────────────────────────────────────────
    ledcSetup(PWM_CH_A, PWM_FREQ_HZ, PWM_RES_BITS);
    ledcAttachPin(PWMA, PWM_CH_A);

    ledcSetup(PWM_CH_B, PWM_FREQ_HZ, PWM_RES_BITS);
    ledcAttachPin(PWMB, PWM_CH_B);

    driverEnable(true);

    Serial.begin(115200);
    // Note: esp_random() is a hardware TRNG and needs no seeding.

    // ── SENSOR CALIBRATION ────────────────────────────────────
    // Slowly sweep the robot (or a card) across the line & background
    // for ~4 s.  The code records the min/max of every sensor.
    Serial.println("=== CALIBRATION — move robot over line+background now ===");

    for (int i = 0; i < 2000; ++i) {
        const int l = analogRead(LEFT_SENSOR);
        const int c = analogRead(CENTER_SENSOR);
        const int r = analogRead(RIGHT_SENSOR);

        minL = min(minL, l);  maxL = max(maxL, l);
        minC = min(minC, c);  maxC = max(maxC, c);
        minR = min(minR, r);  maxR = max(maxR, r);

        delay(2);
    }

    // Guarantee a minimum contrast range so safeMap never divides by zero.
    auto widen = [](int &lo, int &hi, int margin, int cap) {
        if (hi - lo < 50) {
            lo = max(0,   lo - margin);
            hi = min(cap, hi + margin);
        }
    };
    widen(minL, maxL, 25, 4095);
    widen(minC, maxC, 25, 4095);
    widen(minR, maxR, 25, 4095);

    Serial.printf("CAL  L[%d..%d]  C[%d..%d]  R[%d..%d]\n",
                  minL, maxL, minC, maxC, minR, maxR);

    // ── AUTO-TUNE ─────────────────────────────────────────────
    if (AUTO_TUNE_PID) {
        Serial.println("Starting PID auto-tune (relay oscillation)…");
        if (!autoTunePID()) {
            Serial.printf("Auto-tune failed — using defaults: Kp=%.1f Ki=%.1f Kd=%.1f\n",
                          Kp, Ki, Kd);
        }
        delay(200);
    }

    driverEnable(true);  // ensure driver is active before loop()
    Serial.println("=== RUNNING ===");
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
    // ── 1. Sense ──────────────────────────────────────────────
    int L, C, R;
    readSensors(L, C, R);
    const bool detected = (L > DETECT_THRESH || C > DETECT_THRESH || R > DETECT_THRESH);

    if (!detected) {
        // Soft integral decay avoids a burst when the line reappears.
        pidIntegral *= 0.90f;
    }

    const float rawError = getError(L, C, R);
    const float error    = kalmanFilter(rawError);

    // ── 2. PID ────────────────────────────────────────────────
    const float pid = PID(error);

    // ── 3. RL correction ──────────────────────────────────────
    // getState() returns RL_STATE_LOST (5) when !detected, giving the
    // Q-table a dedicated row for recovery rather than polluting the
    // last known positional state with a large negative reward.
    const int   state      = getState(error, detected);
    const int   action     = chooseAction(state);
    const float correction = actions[action];

    // ── 4. Speed profile ──────────────────────────────────────
    // When the line is detected: slow proportionally through tight bends.
    // When the line is lost: clamp to a low creep speed and apply a
    // directional recovery bias from the last known error sign.
    // This prevents the robot driving away while RL searches for the line.
    int   speed;
    float output;
    if (detected) {
        speed  = constrain((int)(baseSpeed - fabsf(error) * 100.0f), 120, 220);
        output = pid + correction;
    } else {
        // Creep slowly; bias steering toward the side the line was last on.
        // pidPrevError retains its sign from getError() even after line loss,
        // so copysignf gives us "turn left if we lost it on the left" etc.
        //
        // Edge case: if the line was lost while nearly centred (|error| < 0.05),
        // pidPrevError is ~0 and copysignf() defaults to +1 → always searches
        // right.  Instead, alternate the search direction every 20 loops so the
        // robot sweeps both sides rather than spiralling away on one side.
        static constexpr int   LOST_SPEED        = 110;
        static constexpr float LOST_STEER        = 40.0f;
        static constexpr float LOST_CENTRE_BAND  = 0.05f;
        static constexpr int   LOST_TOGGLE_LOOPS = 20;   // half-period of sweep

        static int   lostLoopCount  = 0;
        static float lostSearchSign = 1.0f;

        float steerSign;
        if (fabsf(pidPrevError) >= LOST_CENTRE_BAND) {
            // Reliable last-known direction — use it directly.
            steerSign      = (pidPrevError >= 0.0f) ? 1.0f : -1.0f;
            lostLoopCount  = 0;          // reset toggle so it's fresh next time
            lostSearchSign = steerSign;  // seed toggle from last real direction
        } else {
            // Near-zero last error: alternate every LOST_TOGGLE_LOOPS iterations.
            if (++lostLoopCount >= LOST_TOGGLE_LOOPS) {
                lostSearchSign = -lostSearchSign;
                lostLoopCount  = 0;
            }
            steerSign = lostSearchSign;
        }

        speed  = LOST_SPEED;
        output = steerSign * LOST_STEER;
    }

    // ── 5. Actuate ────────────────────────────────────────────
    setMotor(speed + (int)output,
             speed - (int)output);

    // ── 6. Deferred TD update ─────────────────────────────────
    // Apply the Q update from the *previous* iteration now that the
    // robot has physically responded and we have a genuine next-state.
    //   Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',·) − Q(s,a) ]
    //
    // The update runs even when the line is currently lost: the reward
    // stored in rlPrev already encodes –50 for a lost-line transition,
    // so the agent learns that actions leading to line loss are costly.
    // The only hard gate is rlPrev.state >= 0 (no pending update on
    // the very first iteration).
    if (rlPrev.state >= 0) {
        updateQ(rlPrev.state, rlPrev.action, rlPrev.reward, state);
    }

    // Store this iteration's (s, a, r) to be used next time.
    // getReward() returns –50 when !detected, so lost-line transitions
    // are recorded with the correct penalty automatically.
    rlPrev.state  = state;
    rlPrev.action = action;
    rlPrev.reward = getReward(error, detected);
}