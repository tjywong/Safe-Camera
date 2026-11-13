"""Safe lock -- Linux side of the Arduino UNO Q.

Reads the USB webcam, counts fingers with the NumPy FingerNet, runs the same
lock_logic.py as the laptop, and drives the LEDs/servo on the microcontroller
through the Bridge.

Set HARDWARE_TEST = True first to check the button, LEDs, servo and camera.
"""
import time
from pathlib import Path

import cv2
from arduino.app_utils import App, Bridge

import config
from fingernet_numpy import FingerNetNumpy, preprocess
from lock_logic import Lock, State

HARDWARE_TEST = False
CAMERA_INDEX = None   # None = use the first /dev/videoN that gives frames
MIRROR = True         # training photos were mirrored (collect.py), so mirror here too

HERE = Path(__file__).resolve().parent
STATE_CODE = {State.IDLE: 0, State.ENTERING: 1, State.UNLOCKED: 2, State.DENIED: 3}

net = FingerNetNumpy(HERE / "fingernet.npz")
lock = Lock()
button_pressed = False
last_sent = None
cap = None


def on_button():
    # Runs on the Bridge thread: only set a flag (no Bridge.call from here).
    global button_pressed
    button_pressed = True


Bridge.provide("button_pressed", on_button)


def open_camera():
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(10)
    for i in indices:
        c = cv2.VideoCapture(i)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # less USB bandwidth
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # same as on the laptop
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # always use the newest frame
            ok, frame = c.read()
            if ok and frame is not None:
                print(f"camera: /dev/video{i}, {frame.shape[1]}x{frame.shape[0]}")
                return c
        c.release()
    raise RuntimeError("No USB camera found. Is the webcam plugged into the hub? "
                       "Check `ls /dev/video*` on the board.")


def read_frame():
    global cap
    if cap is None:
        cap = open_camera()
    ok, frame = cap.read()
    if not ok:
        print("camera read failed, reopening")
        cap.release()
        cap = None
        return None
    return cv2.flip(frame, 1) if MIRROR else frame


def count_fingers(frame):
    p = net.predict(preprocess(frame, net.img_size))
    return int(p.argmax()), float(p.max())


def send_state():
    global last_sent
    msg = (STATE_CODE[lock.state], len(lock.entered))
    if msg != last_sent:
        try:
            Bridge.call("set_state", msg[0], msg[1], timeout=2)
            last_sent = msg
        except Exception as e:  # keep running; retry next loop
            print("bridge error:", e)


def lock_loop():
    global button_pressed
    frame = read_frame()
    now = time.time()
    if button_pressed:
        button_pressed = False
        lock.press_button(now)
        print("button pressed -> show your code")
    count, prob = None, 0.0
    if frame is not None and lock.state == State.ENTERING:
        count, prob = count_fingers(frame)
    event = lock.update(count, prob, now)
    if event == "digit":
        print(f"digit {len(lock.entered)}/3 accepted")
    elif event:
        print(event)
    send_state()


# ---------------------------------------------------------------- hardware test
test_step, test_next = 0, 0.0
TEST_STEPS = [("both LEDs on (ENTERING)", "state", 1), ("green LED + servo OPEN", "state", 2),
              ("red LED blinking (DENIED)", "state", 3), ("all off + servo LOCKED", "state", 0)]


def test_loop():
    global button_pressed, test_step, test_next
    now = time.time()
    if button_pressed:
        button_pressed = False
        print("BUTTON works")
    if now >= test_next:
        name, _, code = TEST_STEPS[test_step % len(TEST_STEPS)]
        print("test:", name)
        try:
            Bridge.call("set_state", code, 0, timeout=2)
        except Exception as e:
            print("bridge error:", e)
        test_step, test_next = test_step + 1, now + 3
    frame = read_frame()
    if frame is not None:
        t0 = time.time()
        count, prob = count_fingers(frame)
        print(f"camera ok, sees {count} fingers ({prob:.0%}), network {1000 * (time.time() - t0):.0f} ms")
    time.sleep(0.5)


print(f"Safe lock starting. Model input {net.img_size}px. "
      f"{'HARDWARE TEST MODE' if HARDWARE_TEST else 'Lock mode'}")
App.run(user_loop=test_loop if HARDWARE_TEST else lock_loop)
