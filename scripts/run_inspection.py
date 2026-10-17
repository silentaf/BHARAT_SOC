"""Live inspection loop: vision -> severity -> log -> async Llama advisory.

    python scripts/run_inspection.py --model models/screw.pt --source data/mvtec/screw/test
    python scripts/run_inspection.py --model models/pcb.pt --source 0 --show     # camera 0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from qc_assist import PROJECT_ROOT
from qc_assist.advisor import LlamaAdvisor
from qc_assist.anomaly import PatchCore
from qc_assist.config import load_config
from qc_assist.inspection_log import InspectionLog
from qc_assist.io import frames, overlay
from qc_assist.severity import assess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "hailo"])
    ap.add_argument("--source", required=True, help="image folder or camera index")
    ap.add_argument("--config")
    ap.add_argument("--show", action="store_true", help="display annotated frames")
    ap.add_argument("--save-dir", default=str(PROJECT_ROOT / "outputs" / "frames"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    product = cfg["product"]
    model = PatchCore.load(args.model, backend=args.backend)
    log = InspectionLog(PROJECT_ROOT / cfg["log_db"])
    advisor = LlamaAdvisor.from_config(cfg["llama"], PROJECT_ROOT)
    print(f"advisor: {'Llama via ExecuTorch' if advisor.available else 'FALLBACK (Llama not configured yet)'}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pending = {}

    for name, frame in frames(args.source):
        t0 = time.perf_counter()
        pred = model.predict(frame)
        verdict = assess(pred, **cfg["severity"])
        vision_ms = (time.perf_counter() - t0) * 1000
        row = log.add(product, name, verdict, vision_ms)

        label = f"{verdict.severity.upper()}  score {verdict.score:.2f}  {vision_ms:.0f} ms"
        print(f"[{row}] {Path(name).name}: {label}")
        if verdict.severity != "ok" and not pending:  # one advisory in flight at a time
            pending[row] = advisor.advise_async(product, verdict, recent_defect_rate=log.recent_defect_rate(product))

        for r, fut in list(pending.items()):
            if fut.done():
                adv = fut.result()
                log.set_advisory(r, adv.text, adv.source)
                print(f"    advisory for #{r} ({adv.source}): {adv.text}")
                del pending[r]

        vis = overlay(frame, pred.anomaly_map, label)
        if verdict.severity != "ok":
            cv2.imwrite(str(save_dir / f"{row:06d}_{verdict.severity}.jpg"), vis)
        if args.show:
            cv2.imshow("QC-Assist", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    for r, fut in pending.items():
        adv = fut.result()
        log.set_advisory(r, adv.text, adv.source)
        print(f"    advisory for #{r} ({adv.source}): {adv.text}")


if __name__ == "__main__":
    main()
