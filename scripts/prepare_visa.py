"""Extract the VisA PCB categories and arrange them in MVTec-AD layout.

    python -I scripts/prepare_visa.py --tar data/raw/VisA_20220922.tar --csv data/raw/visa_1cls.csv --out data/visa

Result per category:
    data/visa/pcb1/train/good/*.JPG
    data/visa/pcb1/test/{good,bad}/*.JPG
    data/visa/pcb1/ground_truth/bad/<stem>_mask.png
    data/visa/pcb1/defect_labels.csv        image,label  (defect type(s) of every test/bad image)

Uses the official one-class split (split_csv/1cls.csv from amazon-science/spot-diff).
VisA is released under CC BY 4.0.
"""

import argparse
import csv
import io
import shutil
import tarfile
from pathlib import Path

DEFAULT_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    ap.add_argument("--labels-only", action="store_true", help="only (re)write defect_labels.csv")
    args = ap.parse_args()

    out = Path(args.out)
    write_defect_labels(args.tar, args.categories, out)
    if args.labels_only:
        return
    staging = out / "_extracted"
    wanted = set(args.categories)

    with tarfile.open(args.tar) as tar:
        members = []
        for m in tar:
            parts = Path(m.name).parts
            # archive paths look like "<category>/Data/..." possibly under a top-level folder
            if any(p in wanted for p in parts[:2]) and (m.isfile() or m.isdir()):
                members.append(m)
        print(f"extracting {len(members)} entries for {sorted(wanted)}")
        tar.extractall(staging, members=members, filter="data")

    roots = {p.parent for p in staging.rglob("Data") if p.parent.name in wanted}
    root_of = {r.name: r.parent for r in roots}

    counts = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            cat = row["object"]
            if cat not in wanted:
                continue
            base = root_of[cat]
            src = base / row["image"]
            label = "good" if row["label"] == "normal" else "bad"
            dest_dir = out / cat / row["split"] / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dest_dir / src.name)
            if row["mask"]:
                mdir = out / cat / "ground_truth" / label
                mdir.mkdir(parents=True, exist_ok=True)
                shutil.move(base / row["mask"], mdir / f"{src.stem}_mask.png")
            key = (cat, row["split"], label)
            counts[key] = counts.get(key, 0) + 1

    shutil.rmtree(staging)
    for (cat, split, label), n in sorted(counts.items()):
        print(f"{cat:6s} {split:5s} {label:4s} {n}")


def write_defect_labels(tar_path: str, categories, out: Path) -> None:
    """VisA's per-category image_anno.csv names each anomaly's type(s), e.g. "melt,scratch".
    Images keep their file names in our layout, so the label file maps name -> types."""
    with tarfile.open(tar_path) as tar:
        for m in tar:
            parts = Path(m.name).parts
            if m.isfile() and parts[-1] == "image_anno.csv" and parts[-2] in categories:
                rows = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8")))
                (out / parts[-2]).mkdir(parents=True, exist_ok=True)
                with open(out / parts[-2] / "defect_labels.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["image", "label"])
                    for r in rows:
                        if r["label"] != "normal":
                            w.writerow([Path(r["image"]).name, r["label"]])
                print(f"{parts[-2]}: defect_labels.csv ({sum(r['label'] != 'normal' for r in rows)} defects)")


if __name__ == "__main__":
    main()
