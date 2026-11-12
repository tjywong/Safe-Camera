"""Camera helpers + a picker that makes sure we use the USB webcam.

Run `python camera.py` once with the webcam plugged in. It shows every camera
OpenCV can open, side by side; press the number of the USB webcam to save it.
The FaceTime camera and iPhone Continuity Camera are not allowed for this
assignment, so do NOT pick those.
"""
import json
import subprocess

import cv2
import numpy as np
import torch

import config

FORBIDDEN_NAME_HINTS = ("facetime", "iphone", "continuity", "desk view")


def system_camera_names():
    """Names macOS reports (for information only -- order may not match OpenCV)."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=10).stdout
        return [c.get("_name", "?") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


def _open(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def open_camera():
    """Open the webcam chosen with `python camera.py`."""
    if not config.CAMERA_FILE.exists():
        raise SystemExit("No camera chosen yet. Plug in the USB webcam and run: python camera.py")
    index = json.loads(config.CAMERA_FILE.read_text())["index"]
    cap = _open(index)
    if cap is None:
        raise SystemExit(f"Could not open camera {index}. Is the webcam plugged in? "
                         "Re-run `python camera.py` if the index changed.")
    return cap


def center_square(frame):
    """The hand goes in the centre square of the frame; return that crop + its box."""
    h, w = frame.shape[:2]
    s = int(min(h, w) * 0.9)
    y, x = (h - s) // 2, (w - s) // 2
    return frame[y:y + s, x:x + s], (x, y, s)


def to_tensor(rgb_uint8, size=config.IMG_SIZE):
    """HxWx3 uint8 RGB (any size) -> 3 x size x size float tensor in [-1, 1].

    Used by training (validation), the live demo, the lock and the UNO Q, so
    the network always sees exactly the same preprocessing.
    """
    img = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5


def frame_to_input(frame_bgr, size=config.IMG_SIZE):
    """Full camera frame (BGR) -> 1x3xHxW model input."""
    crop, _ = center_square(frame_bgr)
    return to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), size).unsqueeze(0)


def draw_box(frame, color=(255, 255, 255)):
    _, (x, y, s) = center_square(frame)
    cv2.rectangle(frame, (x, y), (x + s, y + s), color, 2)


def main():
    names = system_camera_names()
    print("macOS sees these cameras:", names or "(unknown)")
    if not any(not any(h in n.lower() for h in FORBIDDEN_NAME_HINTS) for n in names):
        print("WARNING: no external webcam detected -- only built-in / iPhone cameras. "
              "Plug in the USB webcam first.")

    caps = {}
    for i in range(6):
        cap = _open(i)
        if cap is not None and cap.read()[0]:
            caps[i] = cap
        elif cap is not None:
            cap.release()
    if not caps:
        raise SystemExit("OpenCV could not open any camera (check camera permission for your terminal/VS Code).")

    print("Press the index number shown on the USB webcam's tile. q = quit without saving.")
    while True:
        tiles = []
        for i, cap in caps.items():
            ok, f = cap.read()
            f = cv2.resize(f, (320, 240)) if ok else np.zeros((240, 320, 3), np.uint8)
            cv2.putText(f, f"[{i}]", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            tiles.append(f)
        cv2.imshow("Pick the USB webcam", np.hstack(tiles))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if ord("0") <= k <= ord("9") and (k - ord("0")) in caps:
            idx = k - ord("0")
            config.CAMERA_FILE.write_text(json.dumps({"index": idx}))
            print(f"Saved camera {idx} to {config.CAMERA_FILE.name}")
            break
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
