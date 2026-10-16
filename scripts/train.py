"""Build the anomaly model from good parts only.

    python scripts/train.py --data data/mvtec/screw --out models/screw.pt

--data is an MVTec-AD-style category folder (uses <data>/train/good) or any
folder of defect-free images.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_assist.anomaly import DetectorConfig, PatchCore, resize_input
from qc_assist.config import load_config
from qc_assist.io import list_images, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--max-train", type=int, help="cap the number of good images for the memory bank")
    ap.add_argument("--n-calib", type=int, default=200,
                    help="unused good images reserved to calibrate the decision threshold")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override detector settings, e.g. --set image_size=320 coreset_ratio=0.05")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = cfg["detector"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        det[k] = type(det[k])(v) if k in det and not isinstance(det[k], list) else v
    det["layers"] = tuple(det["layers"])

    good_dir = Path(args.data) / "train" / "good"
    if not good_dir.is_dir():
        good_dir = Path(args.data)
    paths = list_images(good_dir)
    if not paths:
        sys.exit(f"no images found in {good_dir}")
    print(f"training on {len(paths)} good images from {good_dir}")

    calib_paths = []
    if args.max_train and len(paths) > args.max_train:
        chosen = paths[:: len(paths) // args.max_train][: args.max_train]
        rest = [p for p in paths if p not in set(chosen)]
        calib_paths = rest[:: max(1, len(rest) // args.n_calib)][: args.n_calib] if args.n_calib else []
        paths = chosen
        print(f"memory bank from {len(paths)} evenly spaced images, threshold from {len(calib_paths)} others")

    def load(ps):  # downscale on load to save RAM
        return [resize_input(load_rgb(p), det["image_size"]) for p in ps]

    t0 = time.perf_counter()
    model = PatchCore(DetectorConfig(**det))
    model.fit(load(paths), calib_images=load(calib_paths) or None)
    model.save(args.out)
    print(f"memory bank {tuple(model.memory_bank.shape)}, threshold {model.threshold:.4f}, "
          f"took {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
