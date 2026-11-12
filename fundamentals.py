"""Neural-network fundamentals, written out by hand.  Run: python fundamentals.py

Part 1  A 2-layer network trained with backpropagation + gradient descent that
        I derive and code myself (no autograd), then checked against PyTorch's
        autograd to prove the hand-derived gradients are right.
Part 2  Width vs depth: what they mean and how they change the parameter count.
Part 3  Convolution and max pooling done with plain loops, checked against
        PyTorch's built-in versions.
"""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Part 1: backpropagation and gradient descent by hand
# --------------------------------------------------------------------------- #
def spiral(n_per_class=100, classes=3):
    """Toy 2-D dataset that is NOT linearly separable (needs a hidden layer)."""
    xs, ys = [], []
    for c in range(classes):
        r = torch.linspace(0.05, 1, n_per_class)
        t = torch.linspace(c * 4, (c + 1) * 4, n_per_class) + 0.2 * torch.randn(n_per_class)
        xs.append(torch.stack([r * torch.sin(t), r * torch.cos(t)], 1))
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(xs), torch.cat(ys)


def forward(X, p):
    """Network: X -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> softmax."""
    z1 = X @ p["W1"] + p["b1"]          # N x H   pre-activation
    h = z1.clamp(min=0)                 # N x H   ReLU
    z2 = h @ p["W2"] + p["b2"]          # N x C   logits
    z2s = z2 - z2.max(1, keepdim=True).values      # (numerically stable softmax)
    probs = z2s.exp() / z2s.exp().sum(1, keepdim=True)
    return z1, h, probs


def loss_fn(probs, y):
    """Cross-entropy: average of -log(probability given to the correct class)."""
    return -probs[torch.arange(len(y)), y].log().mean()


def backward(X, y, p, z1, h, probs):
    """Backpropagation = the chain rule, applied from the loss back to each weight.

    dL/dz2 = (probs - onehot(y)) / N        (softmax + cross-entropy combined)
    dL/dW2 = h^T  @ dL/dz2                  (z2 = h W2 + b2)
    dL/db2 = sum over batch of dL/dz2
    dL/dh  = dL/dz2 @ W2^T                  (send the error back through W2)
    dL/dz1 = dL/dh * [z1 > 0]               (ReLU passes gradient only where it was active)
    dL/dW1 = X^T  @ dL/dz1
    dL/db1 = sum over batch of dL/dz1
    """
    N = len(y)
    dz2 = probs.clone()
    dz2[torch.arange(N), y] -= 1
    dz2 /= N
    dh = dz2 @ p["W2"].T
    dz1 = dh * (z1 > 0).float()
    return {"W2": h.T @ dz2, "b2": dz2.sum(0), "W1": X.T @ dz1, "b1": dz1.sum(0)}


def init_params(n_in, hidden, n_out):
    # He initialisation: keep activations from shrinking/exploding through ReLUs.
    return {"W1": torch.randn(n_in, hidden) * math.sqrt(2 / n_in), "b1": torch.zeros(hidden),
            "W2": torch.randn(hidden, n_out) * math.sqrt(2 / hidden), "b2": torch.zeros(n_out)}


def part1():
    print("=" * 70, "\nPart 1: backprop + gradient descent by hand\n" + "=" * 70)
    X, y = spiral()
    p = init_params(2, 64, 3)

    # Gradient check: our hand-written gradients vs. PyTorch autograd.
    z1, h, probs = forward(X, p)
    mine = backward(X, y, p, z1, h, probs)
    pa = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    loss_fn(forward(X, pa)[2], y).backward()
    for k in p:
        print(f"  grad {k:<3} max |mine - autograd| = {(mine[k] - pa[k].grad).abs().max():.2e}")

    # Gradient descent: repeatedly step every weight downhill.
    lr = 1.0
    for step in range(3001):
        z1, h, probs = forward(X, p)
        grads = backward(X, y, p, z1, h, probs)
        for k in p:
            p[k] -= lr * grads[k]          # w <- w - learning_rate * dL/dw
        if step % 500 == 0:
            acc = (probs.argmax(1) == y).float().mean()
            print(f"  step {step:4d}  loss {loss_fn(probs, y):.3f}  accuracy {acc:.1%}")


# --------------------------------------------------------------------------- #
# Part 2: width vs depth
# --------------------------------------------------------------------------- #
def mlp_params(n_in, widths, n_out):
    sizes = [n_in, *widths, n_out]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def part2():
    print("=" * 70, "\nPart 2: width vs depth\n" + "=" * 70)
    print("  width = neurons (or channels) per layer; depth = number of layers")
    n_in = 64 * 64 * 3
    for name, hidden in [("wide & shallow", [512]), ("narrow & deep", [64, 64, 64, 64]),
                         ("wide & deep", [512, 512, 512])]:
        print(f"  fully connected {name:<15} hidden={hidden!s:<18} params={mlp_params(n_in, hidden, 6):>10,}")
    from model import FingerNet, count_params
    for w, d in [(8, 4), (16, 4), (32, 4), (16, 2), (16, 5)]:
        print(f"  FingerNet CNN   width={w:<3} depth={d}            params={count_params(FingerNet(w, d)):>10,}")
    print("  -> a CNN reuses each small filter across the whole image, so it needs far fewer\n"
          "     weights than a fully connected net that has one weight per input pixel.")


# --------------------------------------------------------------------------- #
# Part 3: convolution and pooling by hand
# --------------------------------------------------------------------------- #
def conv2d_loops(x, w, b):
    """x: C_in x H x W, w: C_out x C_in x k x k. 'valid' conv (no padding), stride 1."""
    c_out, _, k, _ = w.shape
    H, W = x.shape[1] - k + 1, x.shape[2] - k + 1
    out = torch.zeros(c_out, H, W)
    for o in range(c_out):
        for i in range(H):
            for j in range(W):
                out[o, i, j] = (x[:, i:i + k, j:j + k] * w[o]).sum() + b[o]  # dot product of filter and patch
    return out


def maxpool_loops(x, k=2):
    C, H, W = x.shape
    out = torch.zeros(C, H // k, W // k)
    for c in range(C):
        for i in range(H // k):
            for j in range(W // k):
                out[c, i, j] = x[c, i * k:(i + 1) * k, j * k:(j + 1) * k].max()
    return out


def part3():
    print("=" * 70, "\nPart 3: convolution and pooling by hand\n" + "=" * 70)
    img = torch.zeros(1, 8, 8)
    img[0, :, 4:] = 1.0                              # left half dark, right half bright
    edge = torch.tensor([[[[-1., 0., 1.]] * 3]])     # 1x1x3x3 vertical-edge filter
    fmap = conv2d_loops(img, edge, torch.zeros(1))
    print("  image (8x8):\n" + "\n".join("   " + " ".join(f"{v:.0f}" for v in r) for r in img[0].tolist()))
    print("  after vertical-edge 3x3 conv (6x6) -- lights up only at the edge:")
    print("\n".join("   " + " ".join(f"{v:2.0f}" for v in r) for r in fmap[0].tolist()))
    print("  after 2x2 max pool (3x3) -- edge kept, size quartered:")
    print("\n".join("   " + " ".join(f"{v:2.0f}" for v in r) for r in maxpool_loops(fmap)[0].tolist()))

    x, w, b = torch.randn(3, 10, 10), torch.randn(4, 3, 3, 3), torch.randn(4)
    d_conv = (conv2d_loops(x, w, b) - F.conv2d(x[None], w, b)[0]).abs().max()
    d_pool = (maxpool_loops(x) - F.max_pool2d(x[None], 2)[0]).abs().max()
    print(f"  check vs PyTorch: conv max diff {d_conv:.1e}, max-pool max diff {d_pool:.1e}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
