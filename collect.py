"""Collect labelled training images from the USB webcam.

Keys:
  0-5  start a burst for that many fingers (0 = fist or empty frame)
  q    quit

Each burst: 2-second countdown, then BURST_SIZE photos over ~8 seconds.
While it records, MOVE your hand: shift it around the box, tilt it, move it
closer/farther, and use different fingers for the same count (e.g. 2 = index+middle,
or thumb+index). Variety is what makes the network generalise.

Tips for a good dataset (aim for >= 300 images per class):
  * several bursts per class, in different lighting / backgrounds / hands
  * include class 0 with an EMPTY box as well as a fist
"""
import time
from collections import Counter

import cv2

import config
from camera import center_square, draw_box, open_camera

BURST_SIZE = 50
CAPTURE_EVERY = 0.15   # seconds between saved photos
COUNTDOWN = 2.0


def counts():
    return Counter({c: len(list((config.DATA_DIR / str(c)).glob("*.jpg")))
                    for c in range(config.NUM_CLASSES)})


def main():
    for c in range(config.NUM_CLASSES):
        (config.DATA_DIR / str(c)).mkdir(parents=True, exist_ok=True)
    cap = open_camera()
    n = counts()
    label, burst, start, saved, last = None, None, 0.0, 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        crop, _ = center_square(frame)
        now = time.time()
        view = frame.copy()

        if label is not None:
            if now - start < COUNTDOWN:
                msg, color = f"{label} fingers in {COUNTDOWN - (now - start):.1f}s", (0, 200, 255)
            else:
                color = (0, 0, 255)
                if now - last >= CAPTURE_EVERY:
                    img = cv2.resize(crop, (config.SAVE_SIZE, config.SAVE_SIZE), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(config.DATA_DIR / str(label) / f"{burst}_{saved:03d}.jpg"), img)
                    saved, last = saved + 1, now
                    n[label] += 1
                msg = f"REC {label}: {saved}/{BURST_SIZE} - move your hand!"
                if saved >= BURST_SIZE:
                    label = None
            draw_box(view, color)
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            draw_box(view)
            cv2.putText(view, "Press 0-5 to record, q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(view, "  ".join(f"{c}:{n[c]}" for c in range(config.NUM_CLASSES)),
                    (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("collect", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if label is None and ord("0") <= k < ord("0") + config.NUM_CLASSES:
            label = k - ord("0")
            burst = time.strftime("%Y%m%d-%H%M%S")
            start, saved, last = now, 0, 0.0

    cap.release()
    cv2.destroyAllWindows()
    print("Images per class:", dict(sorted(n.items())))


if __name__ == "__main__":
    main()
