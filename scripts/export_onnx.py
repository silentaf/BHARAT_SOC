"""Export the patch-feature network (the NPU part of the pipeline) to ONNX, verify it,
and write a calibration set for the Hailo Dataflow Compiler.

    python scripts/export_onnx.py --model models/pcb1.pt --calib data/visa/pcb1/train/good

Outputs (models/):
  <name>_embedder.onnx   input: 1x3xSxS RGB float in 0..255 (normalisation is inside the graph)
                         output: 1xCxHxW patch features
  <name>_calib.npy       N x S x S x 3 float32 RGB 0..255 — Hailo quantisation calibration data

Then on an x86 Linux machine with the Hailo DFC: see docs/hailo.md.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch

from qc_assist.anomaly import IMAGENET_MEAN, IMAGENET_STD, PatchCore
from qc_assist.io import list_images, load_rgb


class NPUGraph(torch.nn.Module):
    """Normalisation + embedder, so the NPU takes raw camera pixels."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.embedder((x - self.mean) / self.std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", help="folder of good images for quantisation calibration")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    pc = PatchCore.load(args.model)
    size = pc.cfg.image_size
    graph = NPUGraph(pc.embedder).eval()
    stem = Path(args.model).with_suffix("")
    onnx_path = Path(f"{stem}_embedder.onnx")

    dummy = torch.rand(1, 3, size, size) * 255
    torch.onnx.export(graph, dummy, onnx_path, input_names=["image"], output_names=["features"],
                      opset_version=args.opset, dynamo=False)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify: ONNX Runtime output must match PyTorch.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = graph(dummy).numpy()
    out = sess.run(None, {"image": dummy.numpy()})[0]
    print(f"max abs diff vs PyTorch: {np.abs(out - ref).max():.2e}  output shape {out.shape}")

    x = dummy.numpy()
    for _ in range(5):
        sess.run(None, {"image": x})
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"image": x})
    print(f"ONNX Runtime CPU: {(time.perf_counter() - t0) / 50 * 1000:.1f} ms / image (backbone only)")

    if args.calib:
        paths = list_images(args.calib)[: args.n_calib]
        calib = np.stack([cv2.resize(load_rgb(p), (size, size), interpolation=cv2.INTER_AREA) for p in paths])
        calib_path = Path(f"{stem}_calib.npy")
        np.save(calib_path, calib.astype(np.float32))
        print(f"calibration set {calib.shape} -> {calib_path}")


if __name__ == "__main__":
    main()
