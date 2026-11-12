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
// Needs the "Servo" library, version 1.3.0 or newer (UNO Q support).

#include <Arduino_RouterBridge.h>
#include <Servo.h>

const int BUTTON_PIN = 2;
const int GREEN_LED = 5;
const int RED_LED = 6;
const int SERVO_PIN = 9;

// Adjust once the servo is mounted on the latch.
const int LOCKED_ANGLE = 0;
const int OPEN_ANGLE = 90;
// Safety: relock on our own even if Linux stops talking to us.
const unsigned long MAX_OPEN_MS = 10000;

// Must match STATE_CODE in python/main.py
enum { IDLE = 0, ENTERING = 1, UNLOCKED = 2, DENIED = 3 };

Servo latch;
int state = IDLE;
int digits = 0;
unsigned long stateSince = 0;
unsigned long flashUntil = 0;

int lastReading = HIGH;
int stableButton = HIGH;
unsigned long lastChange = 0;

// Called from Linux. provide_safe => runs between loop() passes, so it may
// safely use digitalWrite / Servo.
void set_state(int newState, int newDigits) {
  if (newState == ENTERING && newDigits > digits) {
    flashUntil = millis() + 200;          // LEDs blink off briefly: digit accepted
  }
  if (newState != state) stateSince = millis();
  state = newState;
  digits = newDigits;
  latch.write(state == UNLOCKED ? OPEN_ANGLE : LOCKED_ANGLE);
}

// Called from Linux in hardware-test mode: move the servo to any angle.
void servo_test(int angle) {
  latch.write(constrain(angle, 0, 180));
}

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  latch.attach(SERVO_PIN);
  latch.write(LOCKED_ANGLE);              // locked from the moment it powers up

  digitalWrite(RED_LED, HIGH);            // red = booting, waiting for Linux
  Bridge.begin();                         // blocks until the Linux side is up
  Bridge.provide_safe("set_state", set_state);
  Bridge.provide_safe("servo_test", servo_test);
  digitalWrite(RED_LED, LOW);
}

void loop() {
  unsigned long now = millis();

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
