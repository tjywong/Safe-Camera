"""Export the trained model to ONNX for the UNO Q, and check the export.

ONNX is a portable file format for neural networks. On the UNO Q's Linux side
we can run it with the small `onnxruntime` package instead of installing all
of PyTorch.
"""
import numpy as np
import torch

import config
import model as fingernet


def main():
    net = fingernet.load()
    dummy = torch.randn(1, 3, net.img_size, net.img_size)
    torch.onnx.export(net, dummy, str(config.ONNX_PATH), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"Wrote {config.ONNX_PATH} ({config.ONNX_PATH.stat().st_size / 1024:.0f} KB)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping the comparison check.")
        return
    sess = ort.InferenceSession(str(config.ONNX_PATH))
    x = torch.randn(8, 3, net.img_size, net.img_size)
    with torch.no_grad():
        ref = net(x).numpy()
    out = np.concatenate([sess.run(None, {"image": x[i:i + 1].numpy()})[0] for i in range(len(x))])
    print(f"PyTorch vs ONNX max difference: {np.abs(ref - out).max():.2e}  "
          f"(same predictions: {(ref.argmax(1) == out.argmax(1)).all()})")


if __name__ == "__main__":
    main()
