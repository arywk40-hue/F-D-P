#include <Arduino.h>

#if !defined(ARDUINO_ARCH_ESP32)
#error "This sketch requires an ESP32 board. In Arduino IDE select Tools > Board > ESP32 Dev Module (or your ESP32 board)."
#endif

/*
 * Basic 5-IR line follower for ESP32 + L298N
 * ------------------------------------------------------------
 * What this sketch keeps:
 * - 5 sensor calibration
 * - autonomous PID auto-tuning at boot
 * - clean PID-only line following
 *
 * What this sketch removes:
 * - Q-learning / RL layer
 * - extra experimentation logicc
 *
 * Wiring notes:
 * - Sensors: FL, L, C, R, FR on ADC pins
 * - L298N:
 *   PWMA -> ENA
 *   PWMB -> ENB
 *   AIN1/AIN2 and BIN1/BIN2 -> IN1..IN4
 * - STBY is set to -1 by default because L298N usually does not have it.
 */

// Motor pins
#define PWMA  2
#define AIN1  4
#define AIN2  16

#define PWMB  19
#define BIN1  18
#define BIN2  5

static constexpr int STBY = -1;

// Sensor pins
#define FAR_LEFT_SENSOR  32
#define LEFT_SENSOR      25
#define CENTER_SENSOR    26
#define RIGHT_SENSOR     27
#define FAR_RIGHT_SENSOR 33

static constexpr int NUM_SENSORS = 5;
static const int sensorPins[NUM_SENSORS]    = { FAR_LEFT_SENSOR, LEFT_SENSOR, CENTER_SENSOR, RIGHT_SENSOR, FAR_RIGHT_SENSOR };
static const int sensorWeights[NUM_SENSORS] = { -2, -1, 0, 1, 2 };
static const char * const sensorNames[NUM_SENSORS] = { "FL", "L", "C", "R", "FR" };

// Running behaviour
static int   baseSpeed         = 145;
static constexpr int FOLLOW_MIN_SPEED = 95;
static constexpr int FOLLOW_MAX_SPEED = 220;
static constexpr int LOST_SPEED       = 90;
static constexpr float LOST_STEER     = 50.0f;
static constexpr int DETECT_THRESH    = 300;

// Calibration
static constexpr uint32_t CALIBRATION_TIME_MS = 4000;

// PID fallback values
static float Kp = 45.0f;
static float Ki = 0.0f;
static float Kd = 18.0f;

// Autonomous tuning
static constexpr bool AUTO_TUNE_PID         = true;
static constexpr int  TUNE_BASE_SPEED       = 110;
static constexpr float TUNE_RELAY_AMPLITUDE = 45.0f;
static constexpr float TUNE_DEADBAND        = 0.04f;
static constexpr uint32_t TUNE_TIMEOUT_MS   = 7000;
static constexpr uint32_t TUNE_MIN_OSC_MS   = 250;
static constexpr int TUNE_CYCLES            = 4;
static constexpr float TUNE_GAIN_SCALE      = 0.85f;

// PID state
static float pidPrevError      = 0.0f;
static float pidPrevDerivative = 0.0f;
static float pidIntegral       = 0.0f;
static float lastSeenError     = 0.0f;

// Calibration state
static int sensorMin[NUM_SENSORS] = { 4095, 4095, 4095, 4095, 4095 };
static int sensorMax[NUM_SENSORS] = { 0, 0, 0, 0, 0 };

static int safeMap(int x, int in_min, int in_max, int out_min, int out_max);
static void driverEnable(bool on);
static void stopMotors();
static void setMotor(int leftMotor, int rightMotor);
static void resetPID();
static void readRawSensors(int raw[NUM_SENSORS]);
static void readSensors(int sensors[NUM_SENSORS]);
static bool isLineDetected(const int sensors[NUM_SENSORS]);
static float getError(const int sensors[NUM_SENSORS]);
static void calibrateSensors();
static bool autoTunePID();
static float pidStep(float error);
static void printPidValues(const char *label);

static int safeMap(int x, int in_min, int in_max, int out_min, int out_max) {
    if (in_max <= in_min) return (out_min + out_max) / 2;

    const long scaled = (long)(x - in_min) * (out_max - out_min)
                      / (long)(in_max - in_min) + out_min;
    return (int)constrain(scaled, (long)out_min, (long)out_max);
}

static void driverEnable(bool on) {
    if (STBY >= 0) {
        digitalWrite(STBY, on ? HIGH : LOW);
    }
}

static void stopMotors() {
    analogWrite(PWMA, 0);
    analogWrite(PWMB, 0);
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
    driverEnable(false);
}

static void setMotor(int leftMotor, int rightMotor) {
    leftMotor  = constrain(leftMotor, -255, 255);
    rightMotor = constrain(rightMotor, -255, 255);

    driverEnable(true);

    digitalWrite(AIN1, leftMotor >= 0 ? HIGH : LOW);
    digitalWrite(AIN2, leftMotor <  0 ? HIGH : LOW);
    analogWrite(PWMA, abs(leftMotor));

    digitalWrite(BIN1, rightMotor >= 0 ? HIGH : LOW);
    digitalWrite(BIN2, rightMotor <  0 ? HIGH : LOW);
    analogWrite(PWMB, abs(rightMotor));
}

static void resetPID() {
    pidPrevError      = 0.0f;
    pidPrevDerivative = 0.0f;
    pidIntegral       = 0.0f;
}

static void readRawSensors(int raw[NUM_SENSORS]) {
    for (int i = 0; i < NUM_SENSORS; ++i) {
        raw[i] = analogRead(sensorPins[i]);
    }
}

static void readSensors(int sensors[NUM_SENSORS]) {
    int raw[NUM_SENSORS];
    readRawSensors(raw);

    for (int i = 0; i < NUM_SENSORS; ++i) {
        sensors[i] = safeMap(raw[i], sensorMin[i], sensorMax[i], 0, 1000);
    }
}

static bool isLineDetected(const int sensors[NUM_SENSORS]) {
    for (int i = 0; i < NUM_SENSORS; ++i) {
        if (sensors[i] > DETECT_THRESH) return true;
    }
    return false;
}

static float getError(const int sensors[NUM_SENSORS]) {
    long sum = 0;
    long weightedSum = 0;

    for (int i = 0; i < NUM_SENSORS; ++i) {
        sum += (long)sensors[i];
        weightedSum += (long)sensorWeights[i] * (long)sensors[i];
    }

    if (sum < 50) return lastSeenError;

    const float error = (float)weightedSum / (2.0f * (float)sum);
    return constrain(error, -1.0f, 1.0f);
}

static void calibrateSensors() {
    Serial.println("=== CALIBRATION START ===");
    Serial.println("Move all 5 sensors over both line and background for 4 seconds.");

    const uint32_t start = millis();
    while (millis() - start < CALIBRATION_TIME_MS) {
        int raw[NUM_SENSORS];
        readRawSensors(raw);

        for (int i = 0; i < NUM_SENSORS; ++i) {
            sensorMin[i] = min(sensorMin[i], raw[i]);
            sensorMax[i] = max(sensorMax[i], raw[i]);
        }

        delay(2);
    }

    for (int i = 0; i < NUM_SENSORS; ++i) {
        if (sensorMax[i] - sensorMin[i] < 50) {
            sensorMin[i] = max(0, sensorMin[i] - 25);
            sensorMax[i] = min(4095, sensorMax[i] + 25);
        }
    }

    Serial.print("CAL");
    for (int i = 0; i < NUM_SENSORS; ++i) {
        Serial.print("  ");
        Serial.print(sensorNames[i]);
        Serial.print("[");
        Serial.print(sensorMin[i]);
        Serial.print("..");
        Serial.print(sensorMax[i]);
        Serial.print("]");
    }
    Serial.println();
}

static bool autoTunePID() {
    resetPID();

    Serial.println("=== AUTO PID TUNE START ===");
    Serial.println("Place the robot on the line with the center sensor near the track.");

    float ampBuf[TUNE_CYCLES] = {};
    uint32_t periodBuf[TUNE_CYCLES] = {};
    int cyclesRecorded = 0;

    float peakMax = -1e9f;
    float peakMin =  1e9f;
    float lastError = 0.0f;
    bool driveRight = true;
    uint32_t lastCross = 0;
    const uint32_t start = millis();

    while (millis() - start < TUNE_TIMEOUT_MS) {
        int sensors[NUM_SENSORS];
        readSensors(sensors);

        if (!isLineDetected(sensors)) {
            stopMotors();
            Serial.println("AUTO-TUNE FAILED: line lost.");
            return false;
        }

        const float error = getError(sensors);

        if (error > TUNE_DEADBAND) {
            driveRight = false;
        } else if (error < -TUNE_DEADBAND) {
            driveRight = true;
        }

        const float relay = driveRight ? TUNE_RELAY_AMPLITUDE : -TUNE_RELAY_AMPLITUDE;
        setMotor(TUNE_BASE_SPEED + (int)relay, TUNE_BASE_SPEED - (int)relay);

        if (error > peakMax) peakMax = error;
        if (error < peakMin) peakMin = error;

        const bool crossed = (error >= 0.0f && lastError < 0.0f) ||
                             (error <  0.0f && lastError >= 0.0f);

        if (crossed) {
            const uint32_t now = millis();
            if (lastCross != 0) {
                const uint32_t fullPeriod = (now - lastCross) * 2U;
                if (fullPeriod >= TUNE_MIN_OSC_MS) {
                    const int slot = cyclesRecorded % TUNE_CYCLES;
                    ampBuf[slot] = (peakMax - peakMin) * 0.5f;
                    periodBuf[slot] = fullPeriod;
                    ++cyclesRecorded;
                }
            }

            lastCross = now;
            peakMax = error;
            peakMin = error;

            if (cyclesRecorded >= TUNE_CYCLES) break;
        }

        lastError = error;
        delay(5);
    }

    stopMotors();

    if (cyclesRecorded < TUNE_CYCLES) {
        Serial.print("AUTO-TUNE FAILED: only ");
        Serial.print(cyclesRecorded);
        Serial.print("/");
        Serial.print(TUNE_CYCLES);
        Serial.println(" cycles collected.");
        return false;
    }

    float aSum = 0.0f;
    uint32_t pSum = 0;
    for (int i = 0; i < TUNE_CYCLES; ++i) {
        aSum += ampBuf[i];
        pSum += periodBuf[i];
    }

    const float amplitude = aSum / (float)TUNE_CYCLES;
    const float Pu = ((float)pSum / (float)TUNE_CYCLES) / 1000.0f;

    if (amplitude < 0.02f || Pu <= 0.05f) {
        Serial.println("AUTO-TUNE FAILED: noisy or too-small oscillation.");
        return false;
    }

    const float Ku = (4.0f * TUNE_RELAY_AMPLITUDE) / (3.14159265f * amplitude);

    // Classic relay-feedback PID, slightly softened for safer first runs.
    Kp = 0.60f * Ku * TUNE_GAIN_SCALE;
    Ki = (1.20f * Ku / Pu) * TUNE_GAIN_SCALE;
    Kd = (0.075f * Ku * Pu) * TUNE_GAIN_SCALE;

    resetPID();

    Serial.print("AUTO-TUNE OK: Ku=");
    Serial.print(Ku, 2);
    Serial.print(" Pu=");
    Serial.print(Pu, 3);
    Serial.print(" ");
    printPidValues("PID");
    return true;
}

static float pidStep(float error) {
    if (fabsf(error) < 0.02f) error = 0.0f;

    if (fabsf(error) < 0.60f) {
        pidIntegral += error;
    }
    pidIntegral = constrain(pidIntegral, -200.0f, 200.0f);

    const float derivative = 0.65f * (error - pidPrevError)
                           + 0.35f * pidPrevDerivative;

    const float output = Kp * error + Ki * pidIntegral + Kd * derivative;

    pidPrevError = error;
    pidPrevDerivative = derivative;
    return output;
}

static void printPidValues(const char *label) {
    Serial.print(label);
    Serial.print(": Kp=");
    Serial.print(Kp, 2);
    Serial.print(" Ki=");
    Serial.print(Ki, 2);
    Serial.print(" Kd=");
    Serial.println(Kd, 2);
}

void setup() {
    const int motorDirPins[] = { AIN1, AIN2, BIN1, BIN2 };
    for (int i = 0; i < 4; ++i) {
        pinMode(motorDirPins[i], OUTPUT);
    }
    if (STBY >= 0) {
        pinMode(STBY, OUTPUT);
    }
    pinMode(PWMA, OUTPUT);
    pinMode(PWMB, OUTPUT);

    for (int i = 0; i < NUM_SENSORS; ++i) {
        pinMode(sensorPins[i], INPUT);
    }

    Serial.begin(115200);
    delay(500);

    stopMotors();
    calibrateSensors();

    if (AUTO_TUNE_PID) {
        delay(1000);
        if (!autoTunePID()) {
            printPidValues("Using fallback PID");
        }
    }

    driverEnable(true);
    Serial.println("=== LINE FOLLOW MODE ===");
}

void loop() {
    int sensors[NUM_SENSORS];
    readSensors(sensors);

    const bool detected = isLineDetected(sensors);
    int speed = baseSpeed;
    float steer = 0.0f;

    if (detected) {
        const float error = getError(sensors);
        lastSeenError = error;

        speed = constrain(baseSpeed - (int)(fabsf(error) * 55.0f),
                          FOLLOW_MIN_SPEED, FOLLOW_MAX_SPEED);
        steer = pidStep(error);
    } else {
        static int searchCount = 0;
        static float searchSign = 1.0f;

        pidIntegral *= 0.85f;

        if (fabsf(lastSeenError) >= 0.05f) {
            searchSign = (lastSeenError >= 0.0f) ? 1.0f : -1.0f;
            searchCount = 0;
        } else {
            if (++searchCount >= 20) {
                searchSign = -searchSign;
                searchCount = 0;
            }
        }

        speed = LOST_SPEED;
        steer = searchSign * LOST_STEER;
    }

    setMotor(speed + (int)steer, speed - (int)steer);
    delay(5);
}
