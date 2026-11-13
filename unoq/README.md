# Safe lock on the Arduino UNO Q

```
 ┌──────────── UNO Q ─────────────────────────────────────────────┐
 │  Linux (Qualcomm)                     Microcontroller (STM32)  │
 │  python/main.py                       sketch/sketch.ino        │
 │   webcam → FingerNet (NumPy)           button D2 ──notify──┐   │
 │   → lock_logic.py ──Bridge.call("set_state")──►            │   │
 │                       ◄──────────── "button_pressed" ──────┘   │
 │                                        LEDs D5/D6, servo D9    │
 └────────────────────────────────────────────────────────────────┘
```

The network runs as plain NumPy (`fingernet_numpy.py`), so the board needs only
`numpy` and `opencv-python-headless`. No PyTorch or onnxruntime.

## 1. Build the app on the laptop

Run this after every retrain, combination change or code change in `unoq/safe-lock`:

```bash
python unoq/build_app.py
```

This writes `safe-lock/python/fingernet.npz` and copies `lock_logic.py` and `config.py`
(your `COMBINATION`). It checks that the NumPy network matches PyTorch, and only if that
passes (**OK**) it copies the whole app to `~/ArduinoApps/ArduinoApps/safe_lock`.

**Edit the app here in `unoq/safe-lock`, not in `ArduinoApps/safe_lock`.** The build
overwrites that copy.

## 2. Deploy to the board

Uses the deploy tool in your ArduinoApps folder (USB/adb first, then WiFi):

```bash
cd ~/ArduinoApps/ArduinoApps
python3 tools/deploy.py safe_lock
```

The first run takes a while. The board installs `numpy` + `opencv-python-headless`
and compiles the sketch. The sketch drives the servo directly, with no Servo
library: the Servo library's UNO Q build failed to link. To see more of the app's output:

```bash
python3 tools/deploy.py --cmd "arduino-app-cli app logs /home/arduino/ArduinoApps/safe_lock --tail 40"
```

## 3. Hardware test first

Set `HARDWARE_TEST = True` in `unoq/safe-lock/python/main.py`, then rerun `build_app.py`
and deploy. The logs should show:

- `camera: /dev/videoN ...`, then about twice a second: `sees N fingers (…%), network NN ms`
- LEDs and servo cycling every 3 s: both on → green + servo open → red blinking → off + locked
- `BUTTON works` each time you press the button

Fix anything that fails here before going on. Adjust `LOCKED_ANGLE` / `OPEN_ANGLE` in the
sketch to fit your latch. After each move the servo stops being driven, so it goes quiet;
set `SERVO_HOLD = true` if the latch needs it held actively.

## 4. Run the lock

Set `HARDWARE_TEST = False`, rerun `build_app.py` and deploy again. Press the button: both LEDs light up. Show the three
numbers, pulling your hand out of the box between them. LED and servo behavior:

| Event | LEDs | Servo |
|---|---|---|
| Booting (waiting for Linux) | red solid | locked |
| Button pressed, entering code | both on, blink off briefly per accepted digit | locked |
| Correct code | green | open for 5 s, then relocks |
| Wrong code / 20 s timeout | red blinking 3 s | locked |

As a safety measure, the sketch relocks on its own after 10 s even if Linux crashes.

**Mount the webcam the same way it was during data collection:** same height and angle,
looking down, with your arm entering from the same side. The images are mirrored like on
the laptop (`MIRROR = True`).

## 5. Run on the battery

In App Lab, open the `safe_lock` app and set it to start automatically at boot. Then unplug the laptop and
connect: battery → hub (USB-C PD input) → UNO Q, with the webcam in the hub's USB-A port.

## Troubleshooting

- **`No USB camera found`:** run `ls /dev/video*` over SSH. The board has its own video
  devices too, so the code tries each one. If the webcam works over SSH with
  `python3 -c "import cv2; print(cv2.VideoCapture(0).read()[0])"` but not inside App Lab,
  the app container may not have camera access. Ask on the Arduino forum, or run `main.py` directly.
- **Board resets when the servo moves:** add a 470–1000 µF capacitor across 5V/GND, or
  power the servo separately (shared GND).
- **Digits not accepted:** the network may be slower on the board. Lower `STABLE_FRAMES`
  in `config.py` (for example to 5) and rerun `build_app.py`.
