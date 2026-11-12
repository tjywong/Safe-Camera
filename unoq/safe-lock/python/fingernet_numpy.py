"""FingerNet forward pass in plain NumPy -- no PyTorch or ONNX needed on the UNO Q.

The weights come from the trained PyTorch model (unoq/build_app.py exports
them). BatchNorm is "folded" into the convolution beforehand:

    BN(conv(x)) = gamma * (W*x - mean) / sqrt(var + eps) + beta
                = (W * gamma/sqrt(var+eps)) * x  +  (beta - mean * gamma/sqrt(var+eps))
                =            W'             * x  +              b'

so at inference time every stage is just  conv(W', b') -> ReLU -> 2x2 max pool.
"""
import cv2
import numpy as np


def conv3x3(x, w, b):
    """3x3 convolution, padding 1, stride 1.  x: C x H x W,  w: C_out x C x 3 x 3.

    "im2col": gather the 3x3 neighbourhood of every pixel into a column, so the
    whole convolution becomes ONE matrix multiply (filters @ patches).
    """
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((c, 3, 3, h, wd), dtype=x.dtype)
    for i in range(3):
        for j in range(3):
            cols[:, i, j] = xp[:, i:i + h, j:j + wd]
    out = w.reshape(w.shape[0], -1) @ cols.reshape(c * 9, h * wd)
    return out.reshape(-1, h, wd) + b[:, None, None]


def maxpool2x2(x):
    c, h, w = x.shape
    return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class FingerNetNumpy:
    def __init__(self, path):
        d = np.load(path)
        self.img_size = int(d["img_size"])
        self.stages = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["depth"]))]
        self.fc_w, self.fc_b = d["fc_w"], d["fc_b"]

    def logits(self, x):
        """x: 3 x S x S float32 (from preprocess) -> 6 logits."""
        for w, b in self.stages:
            x = maxpool2x2(np.maximum(conv3x3(x, w, b), 0.0))   # conv -> ReLU -> pool
        features = x.mean(axis=(1, 2))                          # global average pool
        return self.fc_w @ features + self.fc_b                 # linear classifier

    def predict(self, x):
        """Probabilities for 0..5 fingers."""
        return softmax(self.logits(x))


def preprocess(frame_bgr, size):
    """Camera frame -> network input. Must match camera.py on the laptop exactly:
    centre square (90% of the short side) -> RGB -> resize -> scale to [-1, 1]."""
    h, w = frame_bgr.shape[:2]
    s = int(min(h, w) * 0.9)
    y, x = (h - s) // 2, (w - s) // 2
    rgb = cv2.cvtColor(frame_bgr[y:y + s, x:x + s], cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
