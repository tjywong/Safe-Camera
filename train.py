"""Train FingerNet on the images in data/.

    python train.py                       # defaults
    python train.py --width 8 --depth 3   # try a narrower / shallower net

What one training step does (see the loop in `run_epoch`):
  1. FORWARD:  logits = model(images)        -- push a batch through the network
  2. LOSS:     cross-entropy(logits, labels) -- how wrong were we? (-log p(correct))
  3. BACKWARD: loss.backward()               -- backpropagation: the chain rule,
               applied from the loss back through every layer, gives
               dLoss/dWeight for every one of the ~100k weights
  4. STEP:     optimizer.step()              -- gradient descent: nudge every
               weight a little *against* its gradient,
               w <- w - lr * (gradient + momentum + weight decay)
  5. zero_grad(): PyTorch adds gradients up, so clear them before the next batch.

Validation images are held out by *burst* (a whole recording session), not by
single frames: neighbouring frames of one burst are near-duplicates, and testing
on them would make the accuracy look better than it really is.
"""
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config
from camera import to_tensor
from model import FingerNet, count_params


def find_images(data_dir):
    """Return {label: {burst_id: [paths]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for label in range(config.NUM_CLASSES):
        for p in sorted((data_dir / str(label)).glob("*.jpg")):
            out[label][p.stem.rsplit("_", 1)[0]].append(p)
    return out


def split(images, val_fraction, seed=0):
    rng = random.Random(seed)
    train, val = [], []
    for label, bursts in images.items():
        ids = sorted(bursts)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction)) if len(ids) >= 2 else 0
        for i, b in enumerate(ids):
            (val if i < n_val else train).extend((p, label) for p in bursts[b])
        if n_val == 0:  # only one burst: fall back to holding out its last 20% of frames
            ps = bursts[ids[0]]
            k = max(1, int(len(ps) * val_fraction))
            train[:] = [t for t in train if t[0] not in ps[-k:]]
            val.extend((p, label) for p in ps[-k:])
            print(f"  note: class {label} has only one burst -- record more for an honest validation score")
    return train, val


class FingerImages(Dataset):
    def __init__(self, items, augment, size=config.IMG_SIZE, rotate=45):
        self.items, self.size = items, size
        # Data augmentation: random variations of each photo, so the network
        # learns "number of fingers" instead of memorising exact pixels.
        self.aug = v2.Compose([
            # rotate up to +-`rotate` degrees, shift up to 8%, zoom 90-110%
            v2.RandomAffine(degrees=rotate, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(),          # left vs right hand
            v2.RandomApply([v2.GaussianBlur(3)], p=0.2),   # webcam motion / focus blur
            v2.ColorJitter(0.35, 0.35, 0.3, 0.04),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5] * 3, [0.5] * 3),
        ]) if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.aug is None:
            return to_tensor(rgb, self.size), label     # identical to what the lock sees live
        return self.aug(torch.from_numpy(rgb).permute(2, 0, 1)), label


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)            # dropout/batchnorm behave differently when training
    total_loss, correct, n = 0.0, 0, 0
    confusion = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)                    # 1. forward
            loss = loss_fn(logits, labels)            # 2. loss
            if training:
                optimizer.zero_grad()                 # 5. clear old gradients
                loss.backward()                       # 3. backpropagation
                optimizer.step()                      # 4. gradient-descent step
                scheduler.step()                      #    adjust the learning rate
            pred = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (pred == labels).sum().item()
            n += len(labels)
            for t, p in zip(labels.cpu(), pred.cpu()):
                confusion[t, p] += 1
    return total_loss / n, correct / n, confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.MODEL_PATH))
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE, help="pixels the network sees")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--rotate", type=float, default=45, help="max random rotation (degrees) during training")
    args = ap.parse_args()

    images = find_images(Path(args.data))
    missing = [c for c in range(config.NUM_CLASSES) if not images.get(c)]
    if missing:
        raise SystemExit(f"No images for classes {missing}. Run collect.py first.")
    train_items, val_items = split(images, args.val)
    print(f"train {len(train_items)} images, validation {len(val_items)} images, rotation augmentation +-{args.rotate:g} deg")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = FingerNet(width=args.width, depth=args.depth).to(device)
    print(f"FingerNet width={args.width} depth={args.depth}: {count_params(model):,} parameters, input {args.img_size}x{args.img_size}, device={device}")

    train_dl = DataLoader(FingerImages(train_items, True, args.img_size, args.rotate), batch_size=args.batch, shuffle=True, num_workers=2,
                          persistent_workers=True)
    val_dl = DataLoader(FingerImages(val_items, False, args.img_size), batch_size=args.batch, num_workers=2,
                        persistent_workers=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Stochastic gradient descent with momentum: "stochastic" because each step
    # uses a random mini-batch rather than the whole dataset; momentum keeps a
    # running average of past gradients so steps don't zig-zag.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # Learning rate starts high (big steps) and decays smoothly to ~0 (fine-tuning).
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dl))

    history, best = [], -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_dl, loss_fn, device, optimizer, scheduler)
        va_loss, va_acc, confusion = run_epoch(model, val_dl, loss_fn, device)
        history.append((tr_loss, va_loss, tr_acc, va_acc))
        flag = ""
        if va_acc >= best:
            best, flag = va_acc, "  * saved"
            torch.save({"arch": {"width": args.width, "depth": args.depth}, "img_size": args.img_size,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "val_acc": va_acc, "confusion": confusion}, out)
        print(f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.1%}   "
              f"val loss {va_loss:.3f} acc {va_acc:.1%}   {time.time() - t0:.1f}s{flag}")

    best_ckpt = torch.load(out)
    print(f"\nBest validation accuracy {best:.1%}. Confusion matrix (rows = truth, cols = prediction):")
    print("     " + " ".join(f"{c:>4}" for c in range(config.NUM_CLASSES)))
    for r, row in enumerate(best_ckpt["confusion"].tolist()):
        print(f"{r:>4} " + " ".join(f"{v:>4}" for v in row))
    plot(history, out.with_name("training_curves.png"))


def plot(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr_l, va_l, tr_a, va_a = zip(*history)
    ep = range(1, len(history) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(ep, tr_l, label="train"); a.plot(ep, va_l, label="validation"); a.set_title("loss"); a.legend()
    b.plot(ep, tr_a, label="train"); b.plot(ep, va_a, label="validation"); b.set_title("accuracy"); b.legend()
    for ax in (a, b):
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
