"""Safe lock -- Linux side of the Arduino UNO Q.

Reads the USB webcam, counts fingers with the NumPy FingerNet, runs the same
lock_logic.py as the laptop, and drives the LEDs/servo on the microcontroller
through the Bridge. Open http://<board-ip>:7000 to see what the camera sees
and what the network predicts.

Set HARDWARE_TEST = True first to check the button, LEDs, servo and camera.
"""
import base64
import json
import time
from pathlib import Path

import cv2
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import config
from fingernet_numpy import FingerNetNumpy, preprocess
from lock_logic import Lock, State

HARDWARE_TEST = False
SERVO_TEST_ANGLES = []   # in hardware test: swing between these instead of the LED/lock steps ([] = normal test)
CAMERA_INDEX = None   # None = use the first /dev/videoN that gives frames
MIRROR = True         # training photos were mirrored (collect.py), so mirror here too
VIEW_FPS = 8          # camera frames per second sent to the web page
VIEW_WIDTH = 320      # width of those frames in pixels

HERE = Path(__file__).resolve().parent
STATE_CODE = {State.IDLE: 0, State.ENTERING: 1, State.UNLOCKED: 2, State.DENIED: 3,
              State.LOCKED_OUT: 4}
# Wrong-attempt count and lockout survive restarts, so pulling the power doesn't
# end a lockout early. Lives in the app folder on the board (deploy never touches it).
LOCK_STATE_FILE = HERE.parent / "lock_state.json"

net = FingerNetNumpy(HERE / "fingernet.npz")
lock = Lock()
button_press = None   # "short" / "long", set by the Bridge thread
last_sent = None
cap = None
last_view = 0.0
ui = WebUI()


def on_short():
    # Runs on the Bridge thread: only set a flag (no Bridge.call from here).
    global button_press
    button_press = "short"


def on_long():
    global button_press
    button_press = "long"


Bridge.provide("button_short", on_short)
Bridge.provide("button_long", on_long)


def load_lock_state():
    try:
        saved = json.loads(LOCK_STATE_FILE.read_text())
        lock.wrong = int(saved["wrong"])
        now = time.time()
        # Clamp: the board's clock can jump at boot before it syncs.
        left = min(float(saved["locked_until"]) - now, lock.lockout_seconds)
        if left > 0:
            lock.lock_out(now + left, now)
    except (OSError, ValueError, KeyError):
        pass   # first run, or an unreadable file: start fresh


def save_lock_state():
    try:
        locked_until = lock.locked_until if lock.state == State.LOCKED_OUT else 0
        LOCK_STATE_FILE.write_text(json.dumps({"wrong": lock.wrong, "locked_until": locked_until}))
    except OSError as e:
        print("couldn't save lock state:", e)


load_lock_state()


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


def crop_box(frame):
    """The square the network looks at: same as preprocess() (centre, 90% of short side)."""
    h, w = frame.shape[:2]
    s = int(min(h, w) * 0.9)
    return (w - s) // 2, (h - s) // 2, s


def publish_view(frame, probs, now):
    """Send the camera image and the prediction to the web page, VIEW_FPS times a second."""
    global last_view
    if now - last_view < 1 / VIEW_FPS:
        return
    last_view = now
    img = frame.copy()
    x, y, s = crop_box(img)
    cv2.rectangle(img, (x, y), (x + s, y + s), (0, 200, 255), 3)
    h, w = img.shape[:2]
    img = cv2.resize(img, (VIEW_WIDTH, h * VIEW_WIDTH // w), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return
    entering = lock.state == State.ENTERING
    ui.send_message("view", {
        "image": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
        "probs": [round(float(v), 3) for v in probs],
        "confidence": lock.confidence,
        "state": lock.state.value,
        "digits": len(lock.entered),
        "need_clear": lock.need_clear,
        "wrong": lock.wrong,
        "max_wrong": lock.max_wrong,
        "lockout_left": max(0.0, lock.locked_until - now) if lock.state == State.LOCKED_OUT else None,
        "hold": lock.streak / lock.stable_frames if entering and lock.streak_value is not None else 0,
        "time_left": max(0.0, lock.timeout - (now - lock.since)) if entering else None,
    })


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
    global button_press
    frame = read_frame()
    now = time.time()
    if button_press:
        kind, button_press = button_press, None
        result = lock.press_button(now, long_press=(kind == "long"))
        if result == "close":
            print(f"{kind} press -> locked")
        elif result == "ignored":
            print(f"{kind} press ignored: locked out for {lock.locked_until - now:.0f} more s")
        else:   # only the log says which kind: the lock itself behaves the same
            print(f"{kind} press -> show your code" + ("" if kind == "long" else " (will be refused)"))
    probs, count, prob = None, None, 0.0
    if frame is not None:   # predict every frame so the web page is live; the lock only uses it while ENTERING
        probs = net.predict(preprocess(frame, net.img_size))
        count, prob = int(probs.argmax()), float(probs.max())
    before = (lock.wrong, lock.state == State.LOCKED_OUT)
    event = lock.update(count, prob, now)
    if event == "digit":
        print(f"digit {len(lock.entered)}/3 accepted")
    elif event == "lockout":
        print(f"wrong attempt {lock.wrong} in a row -> locked out for {lock.lockout_seconds:.0f} s")
    elif event in ("deny", "timeout"):
        print(f"{event}: wrong attempt {lock.wrong}/{lock.max_wrong}")
    elif event:
        print(event)
    if (lock.wrong, lock.state == State.LOCKED_OUT) != before:
        save_lock_state()
    send_state()
    if frame is not None:
        publish_view(frame, probs, now)


# ---------------------------------------------------------------- hardware test
test_step, test_next = 0, 0.0
TEST_STEPS = [("both LEDs on (ENTERING)", "state", 1), ("green LED + servo OPEN", "state", 2),
              ("red LED on (DENIED)", "state", 3), ("all off + servo LOCKED", "state", 0)]


def test_loop():
    global button_press, test_step, test_next
    now = time.time()
    if button_press:
        print(f"BUTTON works ({button_press} press)")
        button_press = None
    if now >= test_next:
        try:
            if SERVO_TEST_ANGLES:
                angle = SERVO_TEST_ANGLES[test_step % len(SERVO_TEST_ANGLES)]
                print(f"test: servo -> {angle} deg")
                Bridge.call("servo_test", angle, timeout=2)
            else:
                name, _, code = TEST_STEPS[test_step % len(TEST_STEPS)]
                print("test:", name)
                Bridge.call("set_state", code, 0, timeout=2)
        except Exception as e:
            print("bridge error:", e)
        test_step, test_next = test_step + 1, now + 3
    if SERVO_TEST_ANGLES:   # servo only: skip the camera so the log stays readable
        time.sleep(0.2)
        return
    frame = read_frame()
    if frame is not None:
        t0 = time.time()
        count, prob = count_fingers(frame)
        print(f"camera ok, sees {count} fingers ({prob:.0%}), network {1000 * (time.time() - t0):.0f} ms")
    time.sleep(0.5)


print(f"Safe lock starting. Model input {net.img_size}px. "
      f"{'HARDWARE TEST MODE' if HARDWARE_TEST else 'Lock mode'}"
      f"{'. Locked out (resumed after restart)' if lock.state == State.LOCKED_OUT else ''}")
App.run(user_loop=test_loop if HARDWARE_TEST else lock_loop)
