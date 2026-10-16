"""Accuracy + latency benchmark on an MVTec-AD-style test split. Results feed the technical report.

    python scripts/evaluate.py --model models/screw.pt --data data/mvtec/screw

Writes outputs/benchmarks/<model>_<host>.json with image AUROC, pixel AUROC
(if ground-truth masks exist) and per-image latency stats.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from qc_assist import PROJECT_ROOT
from qc_assist.anomaly import PatchCore
from qc_assist.io import list_images, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="category folder containing test/ (and optionally ground_truth/)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (Pi 5 has 4 cores)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    model = PatchCore.load(args.model, backend=args.backend)
    data = Path(args.data)
    paths = list_images(data / "test")
    if not paths:
        sys.exit(f"no test images under {data / 'test'}")

    for p in paths[:args.warmup]:
        model.predict(load_rgb(p))

    labels, scores, latencies = [], [], []
    pix_true, pix_score = [], []
    rng = np.random.default_rng(0)
    for p in paths:
        img = load_rgb(p)
        t0 = time.perf_counter()
        pred = model.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000)
        defect = p.parent.name
        labels.append(int(defect != "good"))
        scores.append(pred.score)

        size = pred.anomaly_map.shape[::-1]
        mask_path = data / "ground_truth" / defect / f"{p.stem}_mask.png"
        if defect == "good":
            mask = np.zeros(pred.anomaly_map.shape, dtype=bool)
        elif mask_path.exists():
            mask = cv2.resize(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_NEAREST) > 0
        else:
            continue
        sample = rng.choice(mask.size, size=min(2000, mask.size), replace=False)  # keep memory bounded
        pix_true.append(mask.ravel()[sample])
        pix_score.append(pred.anomaly_map.ravel()[sample])

    lat = np.array(latencies)
    pred_labels = np.array(scores) > 1.0
    y = np.array(labels)
    result = {
        "model": Path(args.model).name,
        "host": platform.node(),
        "machine": platform.machine(),
        "backend": args.backend,
        "torch_threads": torch.get_num_threads(),
        "n_images": len(paths),
        "image_auroc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None,
        "pixel_auroc": float(roc_auc_score(np.concatenate(pix_true), np.concatenate(pix_score)))
        if pix_true and np.concatenate(pix_true).any() else None,
        "recall_at_threshold": float(pred_labels[y == 1].mean()) if y.any() else None,
        "false_alarm_rate": float(pred_labels[y == 0].mean()) if (y == 0).any() else None,
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95))},
        "fps": float(1000 / lat.mean()),
    }
    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"{Path(args.model).stem}_{args.backend}_{platform.node()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
