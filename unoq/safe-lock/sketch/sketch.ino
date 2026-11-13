// Safe lock -- microcontroller (STM32) side of the Arduino UNO Q.
//
// This side handles the hardware; the Linux side (python/main.py) runs the
// camera, the neural network and the lock logic, and tells this sketch what
// to show via the Bridge.
//
//   D2  push button to GND (internal pull-up)  -> Bridge.notify("button_pressed")
//   D5  green LED (+220 ohm)                    -> unlocked
//   D6  red LED (+220 ohm)                      -> denied / booting
//   D9  SG90 servo signal (servo powered from 5V)
//
// The servo is driven directly (no Servo library): every 20 ms we send a pulse
// 0.5-2.4 ms long; the pulse length sets the angle. After each move we stop
// pulsing so the servo goes quiet (no buzzing, less battery); the SG90's gears
// still hold the latch reasonably well. Set SERVO_HOLD = true to hold actively.

#include <Arduino_RouterBridge.h>

const int BUTTON_PIN = 2;
const int GREEN_LED = 5;
const int RED_LED = 6;
const int SERVO_PIN = 9;

// Adjust once the servo is mounted on the latch. Avoid 0 and 180: many SG90s
// can't quite reach them and buzz against their end stop.
const int LOCKED_ANGLE = 10;
const int OPEN_ANGLE = 100;
const int SERVO_MIN_US = 500;              // pulse length at 0 degrees
const int SERVO_MAX_US = 2400;             // pulse length at 180 degrees
const unsigned long SERVO_MOVE_MS = 800;   // keep pulsing this long after a move
const bool SERVO_HOLD = false;

// Safety: relock on our own even if Linux stops talking to us.
const unsigned long MAX_OPEN_MS = 10000;

// Must match STATE_CODE in python/main.py
enum { IDLE = 0, ENTERING = 1, UNLOCKED = 2, DENIED = 3 };

int state = IDLE;
int digits = 0;
unsigned long stateSince = 0;
unsigned long flashUntil = 0;

int servoAngle = LOCKED_ANGLE;
unsigned long servoUntil = 0;
unsigned long lastPulse = 0;

int lastReading = HIGH;
int stableButton = HIGH;
unsigned long lastChange = 0;

void servoWrite(int angle) {
  servoAngle = constrain(angle, 0, 180);
  servoUntil = millis() + SERVO_MOVE_MS;
}

// Call often (every loop). Sends one pulse every 20 ms while the servo is moving.
void servoUpdate() {
  unsigned long now = millis();
  if (!SERVO_HOLD && (long)(now - servoUntil) > 0) return;   // done moving: stay quiet
  if (now - lastPulse < 20) return;
  lastPulse = now;
  int us = map(servoAngle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(us);
  digitalWrite(SERVO_PIN, LOW);
}

// Called from Linux. provide_safe => runs between loop() passes, so it may
// safely use digitalWrite.
void set_state(int newState, int newDigits) {
  if (newState == ENTERING && newDigits > digits) {
    flashUntil = millis() + 200;          // LEDs blink off briefly: digit accepted
  }
  if (newState != state) stateSince = millis();
  bool wasUnlocked = (state == UNLOCKED);
  state = newState;
  digits = newDigits;
  if ((state == UNLOCKED) != wasUnlocked) {
    servoWrite(state == UNLOCKED ? OPEN_ANGLE : LOCKED_ANGLE);
  }
}

// Called from Linux in hardware-test mode: move the servo to any angle.
void servo_test(int angle) {
  servoWrite(angle);
}

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(SERVO_PIN, OUTPUT);

  // Locked from the moment it powers up.
  servoWrite(LOCKED_ANGLE);
  while ((long)(millis() - servoUntil) < 0) servoUpdate();

  digitalWrite(RED_LED, HIGH);            // red = booting, waiting for Linux
  Bridge.begin();                         // blocks until the Linux side is up
  Bridge.provide_safe("set_state", set_state);
  Bridge.provide_safe("servo_test", servo_test);
  digitalWrite(RED_LED, LOW);
}

void loop() {
  unsigned long now = millis();
  servoUpdate();

  // Debounced button: report each press once.
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) {
    lastChange = now;
    lastReading = reading;
  }
  if (now - lastChange > 30 && reading != stableButton) {
    stableButton = reading;
    if (stableButton == LOW) Bridge.notify("button_pressed");
  }

  if (state == UNLOCKED && now - stateSince > MAX_OPEN_MS) {
    set_state(IDLE, 0);
  }

  bool green = false, red = false;
  switch (state) {
    case ENTERING:  green = red = (now >= flashUntil); break;   // both LEDs on
    case UNLOCKED:  green = true; break;
    case DENIED:    red = (now / 200) % 2; break;                // red blinks
    default:        break;                                       // IDLE: off
  }
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
}
