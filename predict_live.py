"""Live test of the trained model: shows the finger count and all 6 probabilities.

Keys:
  0-5  the model is WRONG right now -> save the next ~3 seconds of frames with
       this correct label (keep holding the pose, move it slightly). These
       "hard examples" go into data/<label>/ and fix the mistake at retrain.
  q    quit
"""
import time

import cv2
import torch

import config
import model as fingernet
from camera import center_square, draw_box, frame_to_input, open_camera

SAVE_FRAMES = 20
SAVE_EVERY = 0.15  # seconds


def main():
    net = fingernet.load()
    cap = open_camera()
    print(__doc__)
    label, burst, saved, last = None, None, 0, 0.0
    total_saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        if label is not None and now - last >= SAVE_EVERY:
            crop, _ = center_square(frame)
            img = cv2.resize(crop, (config.SAVE_SIZE, config.SAVE_SIZE), interpolation=cv2.INTER_AREA)
            (config.DATA_DIR / str(label)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(config.DATA_DIR / str(label) / f"{burst}_{saved:03d}.jpg"), img)
            saved, last, total_saved = saved + 1, now, total_saved + 1
            if saved >= SAVE_FRAMES:
                print(f"saved {SAVE_FRAMES} photos as {label}")
                label = None

        with torch.no_grad():
            p = torch.softmax(net(frame_to_input(frame, net.img_size)), 1)[0]
        view = frame.copy()
        draw_box(view, (0, 0, 255) if label is not None else (255, 255, 255))
        cv2.putText(view, f"{p.argmax().item()} fingers ({p.max():.0%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        for c in range(config.NUM_CLASSES):
            y = 80 + c * 28
            cv2.putText(view, str(c), (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(view, (30, y), (30 + int(200 * p[c]), y + 18), (0, 200, 255), -1)
        msg = (f"SAVING as {label}: {saved}/{SAVE_FRAMES} - hold the pose" if label is not None
               else f"wrong? press the correct number (0-5) to save it   [saved {total_saved}]")
        cv2.putText(view, msg, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if label is not None else (255, 255, 255), 2)
        cv2.imshow("predict", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if label is None and ord("0") <= k < ord("0") + config.NUM_CLASSES:
            label, saved, last = k - ord("0"), 0, 0.0
            burst = "live-" + time.strftime("%Y%m%d-%H%M%S")
    cap.release()
    cv2.destroyAllWindows()
    if total_saved:
        print(f"Saved {total_saved} hard-example photos. Retrain: python train.py --epochs 100")


if __name__ == "__main__":
    main()
