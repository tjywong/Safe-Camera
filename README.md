# Safe Camera — a finger-combination lock

Walk up to the safe, push the button, the LEDs light up, and show three numbers
(1–5) with your fingers under the camera. A convolutional neural network built
from scratch in PyTorch counts the fingers; the right sequence unlocks the safe.

Stage 1 (this code): everything runs on a laptop with a **USB webcam** (not the
built-in FaceTime camera or an iPhone). Stage 2: move the trained model to the
Arduino UNO Q.

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

## Steps

| Step | Command | What it does |
|---|---|---|
| 0 | `python fundamentals.py` | Backprop + gradient descent by hand (verified against autograd), width vs depth, convolution and pooling by hand |
| 1 | `python camera.py` | Choose the USB webcam (saved to `camera.json`) |
| 2 | `python collect.py` | Record labelled photos: press 0–5, then move your hand around while it records |
| 3 | `python train.py` | Train FingerNet; prints a confusion matrix and saves `models/training_curves.png` |
| 4 | `python predict_live.py` | Watch the live finger count and probabilities |
| 5 | `python lock.py` | The lock: SPACE = button |
| 6 | `python export.py` | Export `models/fingernet.onnx` for the UNO Q |

Set the secret combination in `config.py` (`COMBINATION`).

## The network (`model.py`)

```
input 3×64×64
 └ [Conv3×3 → BatchNorm → ReLU → MaxPool2×2] × depth    channels: 16 → 32 → 64 → 128 (width = 16)
 └ global average pool → dropout → Linear(128 → 6)      logits for 0..5 fingers
```

~98k parameters by default. `python model.py` prints every layer's output shape.
`python train.py --width 8 --depth 3` trains a narrower, shallower net for comparison.

## Data tips

- Aim for 300 or more images per class, recorded in several bursts with different lighting,
  positions, distances, and hands.
- Class 0 = a fist **or** an empty box. The lock uses it to mean "hand removed" between digits.
- Validation is split by burst, so the accuracy reflects photos the network has never
  seen anything like. If a class is weak in the confusion matrix, record more of it.

## Lock behaviour (`lock_logic.py`)

A digit counts once the same answer has been held for 8 frames at ≥80% confidence.
After each digit, make a fist or move your hand away. You have 20 s to enter all
three digits. The state machine contains no hardware code, so the UNO Q reuses it unchanged.
