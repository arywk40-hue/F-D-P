## FD-P (ESP32 Arduino Core)

This folder contains an ESP32 Arduino sketch for a 5-sensor line-following robot using:
- Sensor normalization + calibration
- Kalman filtering
- PID control
- A small RL (Q-learning) correction layer
- A 5-channel weighted error calculation (`FL/L/C/R/FR`)

### Files
- `FD-P.ino` — ESP32 Arduino sketch (recommended)
- `LMS.c++` — original source kept for reference

### Notes (ESP32 specifics)
- PWM uses **LEDC** (`ledcSetup/ledcAttachPin/ledcWrite`) instead of `analogWrite()`.
- Ensure your motor driver PWM pins (`PWMA`, `PWMB`) are on GPIOs that support LEDC output.
- During the first ~4 seconds (calibration loop), move the robot over **line and background**.

### Motor driver note (L298N)
- L298N typically uses **ENA/ENB** for enable/PWM and **IN1..IN4** for direction.
- This sketch assumes:
	- `PWMA`/`PWMB` go to **ENA/ENB** (PWM)
	- `AIN1/AIN2` and `BIN1/BIN2` go to the direction inputs
- The code contains an optional `STBY` pin. L298N doesn’t have a real standby pin; set `STBY = -1` in `FD-P.ino` if you didn’t wire any global enable.

### Tuning tips

#### Autonomous PID self-tuning (recommended)
The sketch supports a **boot-time PID auto-tune** (relay oscillation method) that estimates PID gains after sensor calibration.

- Toggle it in `FD-P.ino`:
	- `static constexpr bool AUTO_TUNE_PID = true;`
- When enabled, the robot will intentionally **wiggle/oscillate** around the line for a few seconds to estimate `Ku` and `Pu`, then it sets `Kp/Ki/Kd` automatically.

Safety notes:
- Use a **lower battery / slower base speed** for tuning first.
- Keep the robot on a short test track or lift wheels slightly if needed.
- If your chassis can’t safely oscillate, set `AUTO_TUNE_PID = false` and tune manually.

#### Manual tuning fallback
- If it oscillates: reduce `Kp` or increase `Kd` a bit.
- If it’s sluggish: increase `Kp` slightly.
- If motors don't spin: verify `STBY` wiring and motor driver logic (some use inverted enable).
