"""Start the dashboard for the demo video (docs/demo_script.md).

    python scripts/demo.py inspect   # pcb1: good and defective boards, types, Llama advice
    python scripts/demo.py teach     # pcb4 with no model: "Train from folder" -> inspecting in ~1 s

Builds short image sequences under outputs/demo/ (links to VisA images) and runs
scripts/dashboard.py on http://127.0.0.1:8000. The inspect sequence is curated: good boards
the detector passes and defective boards it catches, one or two per defect type. Accuracy
over the full test set is in the report, not in this sequence.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qc_assist.anomaly import PatchCore  # noqa: E402
from qc_assist.io import list_images, load_rgb  # noqa: E402

DEMO = ROOT / "outputs" / "demo"


def link_sequence(folder: Path, paths: list[Path]) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for i, p in enumerate(paths):
        (folder / f"{i:02d}_{p.parent.name}_{p.name}").symlink_to(p)


def inspect_sequence(cat: str = "pcb1", n_good: int = 6, per_type: int = 2) -> Path:
    data = ROOT / "data" / "visa" / cat
    model = PatchCore.load(ROOT / "models" / f"{cat}.pt")
    goods = [p for p in list_images(data / "test" / "good")[:30] if not model.predict(load_rgb(p)).is_defect][:n_good]
    labels = {r["image"]: r["label"] for r in csv.DictReader(open(data / "defect_labels.csv"))}
    bads, taken = [], {}
    for p in list_images(data / "test" / "bad"):
        kind = labels.get(p.name, "")
        if "," in kind or taken.get(kind, 0) >= per_type:
            continue
        pred = model.predict(load_rgb(p))
        if pred.is_defect and pred.defect_type == kind:
            bads.append(p)
            taken[kind] = taken.get(kind, 0) + 1
    seq = [x for pair in zip(goods, bads) for x in pair] + goods[len(bads):] + bads[len(goods):]
    folder = DEMO / f"{cat}_inspect"
    link_sequence(folder, seq)
    print(f"{folder}: {len(goods)} good, {len(bads)} defective ({taken})")
    return folder


def teach_sequences(cat: str = "pcb4", n_teach: int = 20) -> tuple[Path, Path]:
    data = ROOT / "data" / "visa" / cat
    good = list_images(data / "train" / "good")
    step = max(1, len(good) // n_teach)
    teach = DEMO / f"{cat}_teach_good"
    link_sequence(teach, good[::step][:n_teach])
    test_good, test_bad = list_images(data / "test" / "good"), list_images(data / "test" / "bad")
    stream = DEMO / f"{cat}_stream"
    link_sequence(stream, [x for pair in zip(test_good[:8], test_bad[:8]) for x in pair])
    return teach, stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["inspect", "teach"])
    ap.add_argument("--fps", type=float, default=0.5, help="boards per second (slow enough to talk over)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py"), "--max-fps", str(args.fps), "--loop",
           "--port", str(args.port)]
    if args.scene == "inspect":
        cmd += ["--source", str(inspect_sequence()), "--model", str(ROOT / "models" / "pcb1.pt")]
    else:
        teach, stream = teach_sequences()
        cmd += ["--source", str(stream)]
        print(f'\nIn the dashboard, paste this into "Train from folder":\n  {teach}\n')
    print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
