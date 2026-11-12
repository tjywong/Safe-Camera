"""FingerNet: a small convolutional neural network, built from scratch.

Vocabulary
----------
* DEPTH = how many layers are stacked. Here: `depth` conv "stages", plus the
  final linear classifier. Deeper nets build more abstract features layer by
  layer (edges -> finger segments -> whole fingers -> "three fingers").
* WIDTH = how many units / channels each layer has. Here a stage has
  `width * 2**i` channels (16, 32, 64, 128 by default). Wider layers can learn
  more different features at the same level of abstraction.

One stage = Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
* Convolution: a small 3x3 filter slides over the image computing weighted sums.
  The same weights are reused at every position (weight sharing), so a
  "fingertip detector" works anywhere in the image and needs only 3*3*C_in
  weights per output channel instead of one weight per pixel.
* ReLU: max(0, x). The non-linearity; without it stacked layers collapse into a
  single linear function, no matter how deep.
* Max pooling: keep the largest value in each 2x2 block. Halves height and
  width (4x less compute downstream), makes the features tolerant to small
  shifts, and grows the receptive field so later layers "see" more of the hand.
* BatchNorm: rescales each channel to a stable range; makes training faster and
  less sensitive to the learning rate.

After the last stage, global average pooling turns each channel's feature map
into one number, and a linear layer maps those to 6 scores (logits) for
0..5 fingers. Softmax turns logits into probabilities.
"""
import torch
from torch import nn

import config


class FingerNet(nn.Module):
    def __init__(self, width=16, depth=4, num_classes=config.NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.width, self.depth = width, depth
        layers, c_in = [], 3
        for i in range(depth):
            c_out = width * 2 ** i
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):                      # x: N x 3 x 64 x 64
        return self.classifier(self.pool(self.features(x)))  # N x 6 logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def describe(model, size=config.IMG_SIZE):
    """Print the tensor shape after every layer -- watch pooling halve H and W."""
    x = torch.zeros(1, 3, size, size)
    print(f"{'input':<28} {tuple(x.shape)}")
    for layer in list(model.features) + [model.pool] + list(model.classifier):
        x = layer(x)
        n = sum(p.numel() for p in layer.parameters())
        print(f"{layer.__class__.__name__:<28} {str(tuple(x.shape)):<20} params={n}")
    print(f"total parameters: {count_params(model):,}")


def load(path=config.MODEL_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = FingerNet(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.img_size = ckpt.get("img_size", 64)   # input size it was trained at
    return model.to(device).eval()


if __name__ == "__main__":
    describe(FingerNet().eval())
