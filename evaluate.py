"""How robust is the model? Accuracy on held-out recordings, at different hand rotations.

    python evaluate.py                     # models/fingernet.pt
    python evaluate.py --model other.pt

Uses the same validation recordings train.py held out, rotates every photo
by a fixed angle, and reports accuracy per angle + the confusion matrix at 0 deg.
"""
import argparse

import cv2
import torch

import config
import model as fingernet
from camera import to_tensor
from train import find_images, split

ANGLES = (-60, -40, -20, 0, 20, 40, 60)


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()

    net = fingernet.load(args.model)
    _, val_items = split(find_images(config.DATA_DIR), args.val)
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p, _ in val_items]
    labels = torch.tensor([l for _, l in val_items])

    print(f"{len(val_items)} held-out photos, model input {net.img_size}px")
    for deg in ANGLES:
        x = torch.stack([to_tensor(rotate(im, deg), net.img_size) for im in imgs])
        with torch.no_grad():
            pred = torch.cat([net(b).argmax(1) for b in x.split(256)])
        acc = (pred == labels).float().mean().item()
        print(f"  rotated {deg:+4d} deg: {acc:6.1%}  " + "#" * int(acc * 40))
        if deg == 0:
            conf = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES, dtype=torch.long)
            for t, p in zip(labels, pred):
                conf[t, p] += 1
            confusion = conf
    print("\nConfusion matrix at 0 deg (rows = truth, cols = prediction):")
    print("     " + " ".join(f"{c:>4}" for c in range(config.NUM_CLASSES)))
    for r, row in enumerate(confusion.tolist()):
        print(f"{r:>4} " + " ".join(f"{v:>4}" for v in row))


if __name__ == "__main__":
    main()
