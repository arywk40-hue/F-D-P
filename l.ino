#include <Arduino.h>
#include <esp_system.h>

#define PWMA  2
#define AIN1  4
#define AIN2  16

#define PWMB  19
#define BIN1  18
#define BIN2  5

static constexpr int STBY = 17;

#define LEFT_SENSOR   25
#define CENTER_SENSOR 26
#define RIGHT_SENSOR  27

static constexpr int PWM_FREQ_HZ  = 20000;
static constexpr int PWM_RES_BITS = 8;
static constexpr int PWM_CH_A     = 0;
static constexpr int PWM_CH_B     = 1;

static int   baseSpeed = 180;

static float Kp = 40.0f;
static float Ki =  0.0f;
static float Kd = 25.0f;

static constexpr bool     AUTO_TUNE_PID       = true;
static constexpr int      DETECT_THRESH       = 300;
static constexpr int      TUNE_BASE_SPEED     = 140;
static constexpr float    TUNE_RELAY_AMPLITUDE = 60.0f;
static constexpr float    TUNE_TARGET_ERROR   = 0.0f;
static constexpr float    TUNE_DEADBAND       = 0.06f;
static constexpr uint32_t TUNE_TIMEOUT_MS     = 7000;
static constexpr uint32_t TUNE_MIN_OSC_MS     = 1200;

static constexpr float RL_ALPHA   = 0.10f;
static constexpr float RL_GAMMA   = 0.90f;
static constexpr int   RL_EPSILON = 5;

static constexpr float KALMAN_Q = 0.10f;
static constexpr float KALMAN_R = 0.80f;

static float pidPrevError      = 0.0f;
static float pidPrevDerivative = 0.0f;
static float pidIntegral       = 0.0f;

static float kalmX = 0.0f;
static float kalmP = 1.0f;

static int minL = 4095, maxL = 0;
static int minC = 4095, maxC = 0;
static int minR = 4095, maxR = 0;

static constexpr int RL_STATE_LOST = 5;
static constexpr int RL_NUM_STATES = 6;

static float       Qtable[RL_NUM_STATES][5] = {};
static const float actions[5] = { -30.0f, -10.0f, 0.0f, 10.0f, 30.0f };

struct RLTransition {
    int   state  = -1;
    int   action =  0;
    float reward =  0.0f;
};
static RLTransition rlPrev;

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

static bool autoTunePID() {
    pidPrevError      = 0.0f;
    pidPrevDerivative = 0.0f;
    pidIntegral       = 0.0f;
    kalmX             = 0.0f;
    kalmP             = 1.0f;

    const uint32_t start = millis();

    static constexpr int N_CYCLES = 3;
    float    ampBuf[N_CYCLES]    = {};
    uint32_t periodBuf[N_CYCLES] = {};
    int      cycleIdx            = 0;

    float    peakMax        = -1e9f;
    float    peakMin        =  1e9f;
    int      cyclesRecorded = 0;
    uint32_t lastCross      = 0;
    bool     driveRight     = true;
    float    lastErr        = 0.0f;

    driverEnable(true);

    while (millis() - start < TUNE_TIMEOUT_MS) {
        int L, C, R;
        readSensors(L, C, R);

        if (L < DETECT_THRESH && C < DETECT_THRESH && R < DETECT_THRESH) {
            stopMotors();
            Serial.println("AUTO-TUNE: line lost, aborting.");
            return false;
        }

        float err = kalmanFilter(getError(L, C, R));

        if      (err >  (TUNE_TARGET_ERROR + TUNE_DEADBAND)) driveRight = false;
        else if (err <  (TUNE_TARGET_ERROR - TUNE_DEADBAND)) driveRight = true;

        const float u = driveRight ? TUNE_RELAY_AMPLITUDE : -TUNE_RELAY_AMPLITUDE;
        setMotor(TUNE_BASE_SPEED + (int)u,
                 TUNE_BASE_SPEED - (int)u);

        if (err > peakMax) peakMax = err;
        if (err < peakMin) peakMin = err;

        const bool crossed = (err >= 0.0f && lastErr < 0.0f) ||
                             (err <  0.0f && lastErr >= 0.0f);
        if (crossed) {
            const uint32_t now = millis();
            if (lastCross != 0) {
                const uint32_t halfPeriod = now - lastCross;
                const uint32_t fullPeriod = halfPeriod * 2;
                if (fullPeriod >= TUNE_MIN_OSC_MS) {
                    const int slot  = cycleIdx % N_CYCLES;
                    ampBuf[slot]    = (peakMax - peakMin) * 0.5f;
                    periodBuf[slot] = fullPeriod;
                    cycleIdx++;
                    cyclesRecorded++;
                }
            }
            lastCross = now;
            peakMax   = err;
            peakMin   = err;
            if (cyclesRecorded >= N_CYCLES + 1) break;
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

    float    aSum       = 0.0f;
    uint32_t pSum       = 0;
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
    const float Pu = (float)(pSum / (uint32_t)validCount) / 1000.0f;

    if (a < 0.02f) {
        Serial.println("AUTO-TUNE: averaged amplitude too small.");
        return false;
    }
    if (Pu <= 0.05f) {
        Serial.println("AUTO-TUNE: averaged period too short (noise?).");
        return false;
    }

    const float d  = TUNE_RELAY_AMPLITUDE;
    const float Ku = (4.0f * d) / (3.14159265f * a);

    Kp = 0.60f * Ku;
    Ki = (1.20f * Ku) / Pu;
    Kd = 0.075f * Ku * Pu;

    Serial.printf("AUTO-TUNE OK: a=%.3f (avg %d cycles)  Ku=%.2f  Pu=%.3fs  =>  Kp=%.2f  Ki=%.2f  Kd=%.2f\n",
                  a, validCount, Ku, Pu, Kp, Ki, Kd);
    return true;
}

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

float getError(int L, int C, int R) {
    const long sum = (long)L + (long)C + (long)R;
    if (sum < 50) return pidPrevError;
    const float e = (-1.0f * L + 0.0f * C + 1.0f * R) / (float)sum;
    return constrain(e, -1.0f, 1.0f);
}

float kalmanFilter(float measurement) {
    kalmP += KALMAN_Q;
    const float K = kalmP / (kalmP + KALMAN_R);
    kalmX += K * (measurement - kalmX);
    kalmP *= (1.0f - K);
    return kalmX;
}

float PID(float error) {
    if (fabsf(error) < 0.05f) error = 0.0f;
    if (fabsf(error) < 0.5f) pidIntegral += error;
    pidIntegral = constrain(pidIntegral, -300.0f, 300.0f);
    const float derivative = 0.7f * (error - pidPrevError)
                           + 0.3f * pidPrevDerivative;
    const float output = Kp * error + Ki * pidIntegral + Kd * derivative;
    pidPrevError      = error;
    pidPrevDerivative = derivative;
    return output;
}

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

int getState(float error, bool detected) {
    if (!detected)       return RL_STATE_LOST;
    if (error < -0.50f)  return 0;
    if (error < -0.05f)  return 1;
    if (error <=  0.05f) return 2;
    if (error <   0.50f) return 3;
    return 4;
}

int chooseAction(int state) {
    if ((int)(esp_random() % 100) < RL_EPSILON)
        return (int)(esp_random() % 5);
    int best = 0;
    for (int i = 1; i < 5; ++i) {
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

void setup() {
    for (int pin : {AIN1, AIN2, BIN1, BIN2})
        pinMode(pin, OUTPUT);
    if (STBY >= 0) pinMode(STBY, OUTPUT);

    for (int pin : {LEFT_SENSOR, CENTER_SENSOR, RIGHT_SENSOR})
        pinMode(pin, INPUT);

    ledcSetup(PWM_CH_A, PWM_FREQ_HZ, PWM_RES_BITS);
    ledcAttachPin(PWMA, PWM_CH_A);
    ledcSetup(PWM_CH_B, PWM_FREQ_HZ, PWM_RES_BITS);
    ledcAttachPin(PWMB, PWM_CH_B);

    driverEnable(true);
    Serial.begin(115200);

    Serial.println("=== CALIBRATION: move robot over line+background now ===");
    for (int i = 0; i < 2000; ++i) {
        const int l = analogRead(LEFT_SENSOR);
        const int c = analogRead(CENTER_SENSOR);
        const int r = analogRead(RIGHT_SENSOR);
        minL = min(minL, l);  maxL = max(maxL, l);
        minC = min(minC, c);  maxC = max(maxC, c);
        minR = min(minR, r);  maxR = max(maxR, r);
        delay(2);
    }

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

    if (AUTO_TUNE_PID) {
        Serial.println("Starting PID auto-tune...");
        if (!autoTunePID()) {
            Serial.printf("Auto-tune failed — using defaults: Kp=%.1f Ki=%.1f Kd=%.1f\n",
                          Kp, Ki, Kd);
        }
        delay(200);
    }

    driverEnable(true);
    Serial.println("=== RUNNING ===");
}

void loop() {
    int L, C, R;
    readSensors(L, C, R);
    const bool detected = (L > DETECT_THRESH || C > DETECT_THRESH || R > DETECT_THRESH);

    if (!detected) pidIntegral *= 0.90f;

    const float rawError = getError(L, C, R);
    const float error    = kalmanFilter(rawError);
    const float pid      = PID(error);

    const int   state      = getState(error, detected);
    const int   action     = chooseAction(state);
    const float correction = actions[action];

    int   speed;
    float output;

    if (detected) {
        speed  = constrain((int)(baseSpeed - fabsf(error) * 100.0f), 120, 220);
        output = pid + correction;
    } else {
        static constexpr int   LOST_SPEED        = 110;
        static constexpr float LOST_STEER        = 40.0f;
        static constexpr float LOST_CENTRE_BAND  = 0.05f;
        static constexpr int   LOST_TOGGLE_LOOPS = 20;

        static int   lostLoopCount  = 0;
        static float lostSearchSign = 1.0f;

        float steerSign;
        if (fabsf(pidPrevError) >= LOST_CENTRE_BAND) {
            steerSign      = (pidPrevError >= 0.0f) ? 1.0f : -1.0f;
            lostLoopCount  = 0;
            lostSearchSign = steerSign;
        } else {
            if (++lostLoopCount >= LOST_TOGGLE_LOOPS) {
                lostSearchSign = -lostSearchSign;
                lostLoopCount  = 0;
            }
            steerSign = lostSearchSign;
        }

        speed  = LOST_SPEED;
        output = steerSign * LOST_STEER;
    }

    setMotor(speed + (int)output,
             speed - (int)output);

    if (rlPrev.state >= 0) {
        updateQ(rlPrev.state, rlPrev.action, rlPrev.reward, state);
    }

    rlPrev.state  = state;
    rlPrev.action = action;
    rlPrev.reward = getReward(error, detected);
}