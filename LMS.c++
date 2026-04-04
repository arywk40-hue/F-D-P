// NOTE:
// - For ESP32 (Arduino core) use `FD-P.ino` (uses LEDC PWM and includes Arduino headers).
// - This `.c++` file is kept as a reference copy; your editor may show undefined Arduino symbols.

// ================= MOTOR PINS =================
#define PWMA 2
#define AIN1 4
#define AIN2 16

#define PWMB 19
#define BIN1 18
#define BIN2 5

#define STBY 17  

// ================= IR SENSORS =================
#define LEFT_SENSOR   25
#define CENTER_SENSOR 26
#define RIGHT_SENSOR  27

// ================= MOTOR =================
int baseSpeed = 180;

// ================= PID =================
float Kp = 40;
float Ki = 0.0;
float Kd = 25;

float previousError = 0;
float previousDerivative = 0;
float integral = 0;

// ================= KALMAN =================
float x_est = 0;
float P = 1;
float Q = 0.1;
float R = 0.8;

// ================= RL =================
float Qtable[5][5] = {0};
float actions[5] = {-30, -10, 0, 10, 30};

float alpha = 0.1;
float gamma = 0.9;

// ================= CALIBRATION =================
int minL = 4095, maxL = 0;
int minC = 4095, maxC = 0;
int minR = 4095, maxR = 0;

// ================= FUNCTIONS =================

// ---- UTILS ----
static inline int safeMap(int x, int in_min, int in_max, int out_min, int out_max) {
  // Prevent divide-by-zero and weird inversions if calibration failed.
  if (in_max <= in_min) return (out_min + out_max) / 2;
  long v = (long)(x - in_min) * (out_max - out_min) / (long)(in_max - in_min) + out_min;
  if (v < out_min) v = out_min;
  if (v > out_max) v = out_max;
  return (int)v;
}

// ---- READ + NORMALIZE ----
void readSensors(int &L, int &C, int &R) {
  int l = analogRead(LEFT_SENSOR);
  int c = analogRead(CENTER_SENSOR);
  int r = analogRead(RIGHT_SENSOR);

  // normalize
  L = safeMap(l, minL, maxL, 0, 1000);
  C = safeMap(c, minC, maxC, 0, 1000);
  R = safeMap(r, minR, maxR, 0, 1000);
}

// ---- ERROR (3 SENSOR) ----
float getError(int L, int C, int R) {
  // Weighted centroid error in [-1, 1] using all three sensors.
  // More stable than threshold-only logic and reduces oscillations.
  const float wL = -1.0f, wC = 0.0f, wR = 1.0f;
  long sum = (long)L + (long)C + (long)R;
  if (sum < 50) return previousError; // line likely lost; hold last direction
  float e = (wL * L + wC * C + wR * R) / (float)sum;
  // Clamp to expected range.
  if (e < -1.0f) e = -1.0f;
  if (e > 1.0f) e = 1.0f;
  return e;
}

// ---- KALMAN ----
float kalman(float measurement) {
  P += Q;
  float K = P / (P + R);
  x_est += K * (measurement - x_est);
  P *= (1 - K);
  return x_est;
}

// ---- PID ----
float PID(float error) {

  if (abs(error) < 0.05) error = 0; // deadband

  if (abs(error) < 0.5)
    integral += error;

  integral = constrain(integral, -300, 300);

  float derivative = 0.7 * (error - previousError) + 0.3 * previousDerivative;

  float output = Kp * error + Ki * integral + Kd * derivative;

  previousError = error;
  previousDerivative = derivative;

  return output;
}

// ---- RL STATE ----
int getState(float error) {
  if (error < -0.5) return 0;
  if (error < 0) return 1;
  if (error == 0) return 2;
  if (error < 0.5) return 3;
  return 4;
}

// ---- ACTION ----
int chooseAction(int state) {
  if (random(0, 100) < 5) return random(0, 5);

  int best = 0;
  for (int i = 1; i < 5; i++) {
    if (Qtable[state][i] > Qtable[state][best]) best = i;
  }
  return best;
}

// ---- REWARD ----
float getReward(float error, bool detected) {
  if (!detected) return -50;
  return 10 - (abs(error) * 20);
}

// ---- UPDATE Q ----
void updateQ(int s, int a, float reward, int ns) {
  float maxQ = Qtable[ns][0];
  for (int i = 1; i < 5; i++) {
    if (Qtable[ns][i] > maxQ) maxQ = Qtable[ns][i];
  }

  Qtable[s][a] += alpha * (reward + gamma * maxQ - Qtable[s][a]);
}

// ---- MOTOR ----
void setMotor(int A, int B) {
  A = constrain(A, -255, 255);
  B = constrain(B, -255, 255);

  digitalWrite(AIN1, A >= 0);
  digitalWrite(AIN2, A < 0);
  analogWrite(PWMA, abs(A));

  digitalWrite(BIN1, B >= 0);
  digitalWrite(BIN2, B < 0);
  analogWrite(PWMB, abs(B));
}

// ================= SETUP =================
void setup() {
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);

  pinMode(LEFT_SENSOR, INPUT);
  pinMode(CENTER_SENSOR, INPUT);
  pinMode(RIGHT_SENSOR, INPUT);

  digitalWrite(STBY, HIGH);

  Serial.begin(115200);

  // -------- CALIBRATION --------
  // Note: ensure the robot is moved over line and background during this phase.
  for (int i = 0; i < 2000; i++) {
    int l = analogRead(LEFT_SENSOR);
    int c = analogRead(CENTER_SENSOR);
    int r = analogRead(RIGHT_SENSOR);

    minL = min(minL, l); maxL = max(maxL, l);
    minC = min(minC, c); maxC = max(maxC, c);
    minR = min(minR, r); maxR = max(maxR, r);

    delay(2);
  }

  // If calibration didn't see any variation, widen ranges to avoid invalid maps.
  if (maxL - minL < 50) { minL = max(0, minL - 25); maxL = min(4095, maxL + 25); }
  if (maxC - minC < 50) { minC = max(0, minC - 25); maxC = min(4095, maxC + 25); }
  if (maxR - minR < 50) { minR = max(0, minR - 25); maxR = min(4095, maxR + 25); }
}

// ================= LOOP =================
void loop() {

  int L, C, R;
  readSensors(L, C, R);

  bool detected = (L > 300 || C > 300 || R > 300);

  float rawError = getError(L, C, R);
  float error = kalman(rawError);

  float pid = PID(error);

  // RL
  int state = getState(error);
  int action = chooseAction(state);
  float correction = actions[action];

  float output = pid + correction;

  // DYNAMIC SPEED
  int speed = 200 - (abs(error) * 100);
  speed = constrain(speed, 120, 220);

  int motorA = speed + output;
  int motorB = speed - output;

  setMotor(motorA, motorB);

  // LEARNING (only when stable)
  // Learn on the transition (s,a)->(s',r) using the *next* state.
  // Keep learning conservative to avoid fighting the PID.
  if (detected && abs(error) < 0.9) {
    // Recompute state after action influence is observed (approx by current sensed error).
    // In a real setup you may want to update using error from the next loop iteration.
    int nextState = getState(error);
    float reward = getReward(error, detected);
    updateQ(state, action, reward, nextState);
  }
}