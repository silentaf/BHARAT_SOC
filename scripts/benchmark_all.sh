#!/usr/bin/env bash
# Train + evaluate every VisA PCB category with both CPU backends; results land in
# outputs/benchmarks/*.json and a summary table in outputs/benchmarks/summary.md.
#   bash scripts/benchmark_all.sh            (laptop)
#   THREADS=4 bash scripts/benchmark_all.sh  (Pi 5: 4 cores)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
THREADS_ARG=${THREADS:+--threads $THREADS}

for cat in pcb1 pcb2 pcb3 pcb4; do
  [ -f "models/$cat.pt" ] || $PY scripts/train.py --data "data/visa/$cat" --out "models/$cat.pt" --max-train 200
  [ -f "models/${cat}_embedder.onnx" ] || $PY scripts/export_onnx.py --model "models/$cat.pt" --calib "data/visa/$cat/train/good" >/dev/null 2>&1
  backends="torch onnx"
  # Hailo NPU too when HailoRT and the compiled HEF are on this device (Pi 5 + AI HAT+ 2)
  [ -f "models/hailo/${cat}_embedder.hef" ] && $PY -c "import hailo_platform" 2>/dev/null && backends="$backends hailo"
  for backend in $backends; do
    $PY scripts/evaluate.py --model "models/$cat.pt" --data "data/visa/$cat" --backend $backend $THREADS_ARG >/dev/null
  done
done

$PY - <<'EOF'
import json, glob, platform
rows = [json.load(open(p)) for p in sorted(glob.glob("outputs/benchmarks/pcb*_*.json"))]
lines = ["| Category | Backend | Host | Image AUROC | Pixel AUROC | Recall | False alarms | ms mean / p95 | FPS |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['model'][:-3]} | {r['backend']} | {r['machine']} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | "
                 f"{r['recall_at_threshold']:.2f} | {r['false_alarm_rate']:.2f} | "
                 + ("emulator | - |" if r["backend"] == "hailo-emu" else  # x86 simulation time, not NPU speed
                    f"{r['latency_ms']['mean']:.1f} / {r['latency_ms']['p95']:.1f} | {r['fps']:.1f} |"))
open("outputs/benchmarks/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

# Keep a copy in the repo (outputs/ is git-ignored) as evidence for the report.
mkdir -p "results/$(uname -m)"
cp outputs/benchmarks/pcb*_*.json outputs/benchmarks/summary.md "results/$(uname -m)/"
echo "copied to results/$(uname -m)/"
