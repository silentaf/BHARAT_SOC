"""Train the defect-type classifier for one product and measure it (qc_assist/defect_types.py).

    python scripts/train_types.py --model models/pcb1.pt --data data/visa/pcb1

Needs labelled defect images: <data>/test/bad/*.JPG plus <data>/defect_labels.csv
(image,label; several types separated by commas). Accuracy is measured with repeated
stratified 5-fold cross-validation; then the classifier is fitted on all examples and
stored inside the model file, so the dashboard names defect types from then on.
Writes outputs/benchmarks/<model>_types_<host>.json and a copy in results/<arch>/.
"""

import argparse
import csv
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_assist import PROJECT_ROOT  # noqa: E402
from qc_assist.anomaly import PatchCore  # noqa: E402
from qc_assist.defect_types import DefectTypeClassifier  # noqa: E402
from qc_assist.io import load_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="category folder with test/bad/ and defect_labels.csv")
    ap.add_argument("--labels", help="default: <data>/defect_labels.csv")
    ap.add_argument("--min-examples", type=int, default=5, help="types with fewer examples are left out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    from sklearn.model_selection import StratifiedKFold

    data = Path(args.data)
    labels = {r["image"]: set(r["label"].split(",")) for r in csv.DictReader(open(args.labels or data / "defect_labels.csv"))}
    paths = [p for p in sorted((data / "test" / "bad").iterdir()) if p.name in labels]
    model = PatchCore.load(args.model)
    model.defect_types = None
    desc = np.stack([model.predict(load_rgb(p)).descriptor for p in paths])
    sets = [labels[p.name] for p in paths]

    # One training label per image: its most common type (multi-type images count as correct
    # if the prediction is any of their types).
    freq = Counter(t for s in sets for t in s)
    primary = np.array([max(s, key=lambda t: (freq[t], t)) for s in sets])
    counts = Counter(primary)
    keep = np.array([counts[p] >= args.min_examples for p in primary])
    classes = sorted(c for c in counts if counts[c] >= args.min_examples)
    X, y, S = desc[keep], primary[keep], [s for s, k in zip(sets, keep) if k]

    accs, bal, conf_acc, coverage = [], [], [], []
    confusion = Counter()
    for seed in range(args.repeats):
        pred = np.empty(len(y), dtype=object)
        conf = np.empty(len(y))
        for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(X, y):
            clf = DefectTypeClassifier.fit(X[tr], list(y[tr]))
            for i in te:
                p = clf.probabilities(X[i])
                pred[i], conf[i] = clf.classes[int(p.argmax())], p.max()
        ok = np.array([p in s for p, s in zip(pred, S)])
        accs.append(ok.mean())
        bal.append(np.mean([ok[y == c].mean() for c in classes]))
        sure = conf >= 0.5
        coverage.append(sure.mean())
        conf_acc.append(ok[sure].mean() if sure.any() else float("nan"))
        confusion.update(zip(y, pred))

    majority = max(sum(c in s for s in S) for c in classes) / len(S)
    result = {
        "model": Path(args.model).name, "host": platform.node(), "machine": platform.machine(),
        "n_defects": len(S), "types": {c: counts[c] for c in classes},
        "left_out_types": {c: n for c, n in counts.items() if n < args.min_examples},
        "cv": f"{args.repeats}x stratified {args.folds}-fold",
        "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "balanced_accuracy": float(np.mean(bal)), "majority_baseline": float(majority),
        "confident_coverage": float(np.mean(coverage)), "confident_accuracy": float(np.nanmean(conf_acc)),
        "confusion_true_pred": {f"{t} -> {p}": n for (t, p), n in sorted(confusion.items())},
    }

    model.defect_types = DefectTypeClassifier.fit(X, list(y))
    model.save(args.model)
    result["saved_to"] = Path(args.model).name

    out = PROJECT_ROOT / "outputs" / "benchmarks" / f"{Path(args.model).stem}_types_{platform.node()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    res_dir = PROJECT_ROOT / "results" / platform.machine()
    res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, res_dir / out.name)
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_true_pred"}, indent=2))


if __name__ == "__main__":
    main()
