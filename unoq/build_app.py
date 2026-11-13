"""Prepare the UNO Q app from the trained model.  Run on the laptop:

    python unoq/build_app.py

1. Folds BatchNorm into the conv weights and saves them to
   unoq/safe-lock/python/fingernet.npz
2. Copies lock_logic.py and config.py (your COMBINATION) into the app
3. Checks the NumPy network gives the same answers as PyTorch on real photos,
   and that the NumPy preprocessing matches camera.py
4. If it passes, copies the whole app to ~/ArduinoApps/ArduinoApps/safe_lock,
   ready for:  cd ~/ArduinoApps/ArduinoApps && python3 tools/deploy.py safe_lock

Edit the app HERE (unoq/safe-lock), not in ArduinoApps -- this script
overwrites the copy there.
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "unoq" / "safe-lock"
APP_PY = APP / "python"
ARDUINO_APPS = Path.home() / "ArduinoApps" / "ArduinoApps"
INSTALLED = ARDUINO_APPS / "safe_lock"
sys.path.insert(0, str(ROOT))
sys.path.append(str(APP_PY))   # after ROOT, so `import config` gets the project's config.py

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import model as fingernet  # noqa: E402
from camera import frame_to_input, to_tensor  # noqa: E402


def fold(net):
    """Return {name: array} with BatchNorm folded into each conv."""
    out = {"img_size": np.array(net.img_size), "depth": np.array(net.depth)}
    convs = [m for m in net.features if isinstance(m, torch.nn.Conv2d)]
    bns = [m for m in net.features if isinstance(m, torch.nn.BatchNorm2d)]
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        out[f"w{i}"] = (conv.weight * scale[:, None, None, None]).detach().numpy().astype(np.float32)
        out[f"b{i}"] = (bn.bias - bn.running_mean * scale).detach().numpy().astype(np.float32)
    fc = net.classifier[-1]
    out["fc_w"] = fc.weight.detach().numpy().astype(np.float32)
    out["fc_b"] = fc.bias.detach().numpy().astype(np.float32)
    return out


def main():
    net = fingernet.load()
    np.savez(APP_PY / "fingernet.npz", **fold(net))
    for f in ("lock_logic.py", "config.py"):
        shutil.copy(ROOT / f, APP_PY / f)
    print(f"Wrote {APP_PY / 'fingernet.npz'}  (input {net.img_size}px)")
    print(f"Copied lock_logic.py and config.py  (combination {config.COMBINATION})")

    from fingernet_numpy import FingerNetNumpy, preprocess
    npnet = FingerNetNumpy(APP_PY / "fingernet.npz")

    # 1. same answers as PyTorch on real photos
    paths = sorted(config.DATA_DIR.glob("*/*.jpg"))[::25]
    agree, max_diff, t = 0, 0.0, 0.0
    for p in paths:
        x = to_tensor(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), net.img_size)
        with torch.no_grad():
            ref = net(x[None])[0].numpy()
        t0 = time.perf_counter()
        mine = npnet.logits(x.numpy())
        t += time.perf_counter() - t0
        agree += ref.argmax() == mine.argmax()
        max_diff = max(max_diff, float(np.abs(ref - mine).max()))
    print(f"NumPy vs PyTorch on {len(paths)} photos: same answer {agree}/{len(paths)}, "
          f"max logit difference {max_diff:.1e}, {1000 * t / len(paths):.1f} ms/image on this laptop")

    # 2. same preprocessing as the laptop
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    d = np.abs(preprocess(frame, net.img_size) - frame_to_input(frame, net.img_size)[0].numpy()).max()
    print(f"Preprocessing difference vs camera.py: {d:.1e}")
    ok = agree == len(paths) and max_diff < 1e-3 and d < 1e-5
    if not ok:
        sys.exit("MISMATCH - not copying the app. Do not deploy.")
    print("OK")

    if not ARDUINO_APPS.is_dir():
        print(f"{ARDUINO_APPS} not found - copy unoq/safe-lock to the board yourself.")
        return
    # Copy over (not delete-then-copy) so hidden App Lab folders like .cache survive.
    shutil.copytree(APP, INSTALLED, ignore=shutil.ignore_patterns("__pycache__", ".*"), dirs_exist_ok=True)
    print(f"Installed app to {INSTALLED}")
    print(f"Deploy:  cd {ARDUINO_APPS} && python3 tools/deploy.py safe_lock")


if __name__ == "__main__":
    main()
