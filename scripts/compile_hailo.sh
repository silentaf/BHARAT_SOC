#!/usr/bin/env bash
# Compile the NPU part of each model for the Hailo NPU, then measure its INT8 accuracy with the
# Dataflow Compiler's emulator. Runs on x86 Linux with the DFC in .venv-hailo (docs/hailo.md).
#   bash scripts/compile_hailo.sh                  # pcb1..pcb4, HW_ARCH=hailo10h (AI HAT+ 2, DFC 5.x)
#   HW_ARCH=hailo8l bash scripts/compile_hailo.sh  # AI HAT+ 13 TOPS: needs DFC 3.x instead
# Outputs: models/hailo/<name>_embedder.hef (+ .har files), outputs/benchmarks/<name>_hailo-emu_*.json
set -euo pipefail
cd "$(dirname "$0")/.."
HW_ARCH=${HW_ARCH:-hailo10h}
MODELS=${MODELS:-"pcb1 pcb2 pcb3 pcb4"}
ROOT=$PWD
HAILO=$ROOT/.venv-hailo/bin/hailo
export CUDA_VISIBLE_DEVICES=""   # CPU only: leave the GPU to other work on this machine
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p models/hailo

for m in $MODELS; do
  [ -f "models/${m}_embedder.onnx" ] || .venv/bin/python scripts/export_onnx.py --model "models/$m.pt" --calib "data/visa/$m/train/good"
  (
    cd models/hailo
    "$HAILO" parser onnx "../${m}_embedder.onnx" --hw-arch "$HW_ARCH" -y > "${m}_parse.log" 2>&1
    "$HAILO" optimize "${m}_embedder.har" --calib-set-path "../${m}_calib.npy" --hw-arch "$HW_ARCH" > "${m}_optimize.log" 2>&1
    "$HAILO" compiler "${m}_embedder_optimized.har" --hw-arch "$HW_ARCH" > "${m}_compile.log" 2>&1
  )
  echo "$m: $(ls -la "models/hailo/${m}_embedder.hef" | awk '{print $5}') bytes HEF for $HW_ARCH"
  .venv-hailo/bin/python scripts/evaluate.py --model "models/$m.pt" --data "data/visa/$m" --backend hailo-emu 2>/dev/null \
    | grep -E '"(image_auroc|pixel_auroc|recall_at_threshold|false_alarm_rate)"'
done

mkdir -p "results/$(uname -m)"
cp outputs/benchmarks/pcb*_hailo-emu_*.json "results/$(uname -m)/"
