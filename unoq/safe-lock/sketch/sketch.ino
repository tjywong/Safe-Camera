// Safe lock -- microcontroller (STM32) side of the Arduino UNO Q.
//
// This side handles the hardware; the Linux side (python/main.py) runs the
// camera, the neural network and the lock logic, and tells this sketch what
// to show via the Bridge.
//
//   D2  push button to GND (internal pull-up)
//       held >= LONG_PRESS_MS -> Bridge.notify("button_long")  (code can open the lock)
//       released sooner       -> Bridge.notify("button_short") (attempt always ends as wrong)
//   D5  green LED (+220 ohm)                    -> unlocked
//   D6  red LED (+220 ohm)                      -> denied / booting; blinks when locked out
//   Both LEDs flash when a press starts code entry, and on-then-off once per
//   registered digit (right or wrong). Only after the 3rd digit's flash does the
//   lock show green + open, or red.
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

// Set on the mounted latch with the hardware test. If the servo buzzes at 0
// (pushing against its end stop), try 3-5.
const int LOCKED_ANGLE = 0;
const int OPEN_ANGLE = 90;
const int SERVO_MIN_US = 500;              // pulse length at 0 degrees
const int SERVO_MAX_US = 2400;             // pulse length at 180 degrees
const unsigned long SERVO_MOVE_MS = 800;   // keep pulsing this long after a move
const bool SERVO_HOLD = false;
const unsigned long LONG_PRESS_MS = 1000;  // hold this long for a long press
const unsigned long PRESS_FLASH_MS = 400;  // both LEDs on: button press started entry
const unsigned long DIGIT_ON_MS = 300;     // both LEDs on, then...
const unsigned long DIGIT_OFF_MS = 300;    // ...both off: a digit was registered

// Must match STATE_CODE in python/main.py. LOCKED_OUT = too many wrong
// attempts: the button is ignored for a while.
enum { IDLE = 0, ENTERING = 1, UNLOCKED = 2, DENIED = 3, LOCKED_OUT = 4 };

int state = IDLE;
int digits = 0;
unsigned long flashOnUntil = 0;   // both LEDs forced on until here,
unsigned long flashOffUntil = 0;  // then forced off until here
int pendingAngle = -1;            // servo move waiting for the digit flash to finish
int testLeds = -1;                // led_test override: bit 0 green, bit 1 red; -1 = off

int servoAngle = LOCKED_ANGLE;
unsigned long servoUntil = 0;
unsigned long lastPulse = 0;

int lastReading = HIGH;
int stableButton = HIGH;
unsigned long lastChange = 0;
unsigned long pressStart = 0;
bool longSent = false;            // this press already reported as long

void servoWrite(int angle) {
  servoAngle = constrain(angle, 0, 180);
  servoUntil = millis() + SERVO_MOVE_MS;
}

bool startsEntry() {
  return state != UNLOCKED && state != LOCKED_OUT;
}

void flash(unsigned long onMs, unsigned long offMs) {
  flashOnUntil = millis() + onMs;
  flashOffUntil = flashOnUntil + offMs;
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
// safely use digitalWrite. The lock only opens or closes from here: it stays
// open until a button press makes Linux send IDLE.
void set_state(int newState, int newDigits) {
  // Same flash for every registered digit, the 3rd included: it says nothing
  // about whether the digit was right.
  if (newDigits > digits) flash(DIGIT_ON_MS, DIGIT_OFF_MS);
  testLeds = -1;
  bool wasUnlocked = (state == UNLOCKED);
  state = newState;
  digits = newDigits;
  if (state == UNLOCKED && !wasUnlocked) {
    pendingAngle = OPEN_ANGLE;            // open once the flash is over
  } else if (state != UNLOCKED && wasUnlocked) {
    pendingAngle = -1;
    servoWrite(LOCKED_ANGLE);             // close right away
  }
}

// Called from Linux in hardware-test mode: move the servo to any angle.
void servo_test(int angle) {
  servoWrite(angle);
}

// Called from Linux for testing: set the LEDs directly (servo untouched) until
// the next set_state.
void led_test(int green, int red) {
  testLeds = (green ? 1 : 0) | (red ? 2 : 0);
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
  Bridge.provide_safe("led_test", led_test);
  digitalWrite(RED_LED, LOW);
}

void loop() {
  unsigned long now = millis();
  servoUpdate();

  // Debounced button: each press is reported once, as long (as soon as it has
  // been held LONG_PRESS_MS) or short (on release before that).
  // A press starts code entry unless it closes an open lock or comes during a
  // lockout: flash right away rather than waiting for Linux to answer.
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) {
    lastChange = now;
    lastReading = reading;
  }
  if (now - lastChange > 30 && reading != stableButton) {
    stableButton = reading;
    if (stableButton == LOW) {
      pressStart = now;
      longSent = false;
    } else if (!longSent) {
      Bridge.notify("button_short");
      if (startsEntry()) flash(PRESS_FLASH_MS, 0);
    }
  }
  if (stableButton == LOW && !longSent && now - pressStart >= LONG_PRESS_MS) {
    longSent = true;
    Bridge.notify("button_long");
    if (startsEntry()) flash(PRESS_FLASH_MS, 0);
  }

  bool flashing = (long)(now - flashOffUntil) < 0;
  if (pendingAngle >= 0 && !flashing) {
    servoWrite(pendingAngle);
    pendingAngle = -1;
  }

  bool green = false, red = false;
  if ((long)(now - flashOnUntil) < 0) {
    green = red = true;                                          // flash: both on
  } else if (flashing) {
                                                                 // flash: both off
  } else if (state == UNLOCKED) {
    green = true;
  } else if (state == DENIED) {
    red = true;                                                  // stays until a press
  } else if (state == LOCKED_OUT) {
    red = (now / 250) % 2;                                       // blinks for the lockout
  }                                                              // IDLE / ENTERING: off
  if (testLeds >= 0) {
    green = testLeds & 1;
    red = testLeds & 2;
  }
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
}
