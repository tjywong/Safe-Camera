"""The safe lock, running on the laptop with the USB webcam.

  SPACE = long press on the safe's button     s = short press     q = quit

Press SPACE: the (on-screen) LEDs light up. Show the three numbers one at a
time, making a fist or pulling your hand away between numbers. Only a long
press can open the lock: after a short press the attempt always ends as wrong.
Three wrong attempts in a row lock the button out for LOCKOUT_SECONDS.
"""
import time

import cv2
import numpy as np
import torch

import config
import model as fingernet
from camera import draw_box, frame_to_input, open_camera
from lock_logic import Lock, State

PANEL_W = 260
COLORS = {State.IDLE: (120, 120, 120), State.ENTERING: (0, 200, 255),
          State.UNLOCKED: (0, 200, 0), State.DENIED: (0, 0, 230), State.LOCKED_OUT: (0, 0, 160)}
TEXT = {State.IDLE: "LOCKED - press button", State.ENTERING: "Enter code...",
        State.UNLOCKED: "UNLOCKED", State.DENIED: "ACCESS DENIED", State.LOCKED_OUT: "LOCKED OUT"}


def draw_panel(h, lock, count, prob):
    panel = np.zeros((h, PANEL_W, 3), np.uint8)
    color = COLORS[lock.state]
    cv2.putText(panel, TEXT[lock.state], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    # Three "LEDs": dark when idle, lit when entering; filled in as each digit is accepted.
    for i in range(3):
        center = (50 + i * 80, 110)
        if lock.state == State.ENTERING:
            filled = i < len(lock.entered)
            cv2.circle(panel, center, 25, (0, 220, 255) if filled else (60, 140, 160), -1)
            cv2.circle(panel, center, 25, (255, 255, 255), 2)
        else:
            cv2.circle(panel, center, 25, color if lock.state != State.IDLE else (50, 50, 50), -1)
    if lock.state == State.ENTERING:
        hint = "make a fist / remove hand" if lock.need_clear else f"show digit {len(lock.entered) + 1}"
        cv2.putText(panel, hint, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        left = config.ENTRY_TIMEOUT - (time.time() - lock.since)
        cv2.putText(panel, f"{left:4.1f}s left", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if count is not None:
            cv2.putText(panel, f"sees: {count}  ({prob:.0%})", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    elif lock.state == State.LOCKED_OUT:
        left = lock.locked_until - time.time()
        cv2.putText(panel, f"try again in {left:3.0f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if lock.wrong and lock.state != State.LOCKED_OUT:
        cv2.putText(panel, f"wrong in a row: {lock.wrong}/{lock.max_wrong}", (10, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 1)
    return panel


def main():
    net = fingernet.load()
    cap = open_camera()
    lock = Lock()
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        count = prob = None
        if lock.state == State.ENTERING:          # only run the network when needed
            with torch.no_grad():
                p = torch.softmax(net(frame_to_input(frame, net.img_size)), 1)[0]
            prob, count = p.max().item(), p.argmax().item()
        event = lock.update(count, prob or 0.0, now)
        if event:
            print(event, lock.entered if event in ("digit", "deny") else "")
        draw_box(frame, COLORS[lock.state])
        cv2.imshow("Safe", np.hstack([frame, draw_panel(frame.shape[0], lock, count, prob)]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord(" "), ord("s")):
            lock.press_button(now, long_press=(k == ord(" ")))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
