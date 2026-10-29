#!/usr/bin/env bash
# Pack what an Arm device needs besides the code (git clone gives the code): the exported Llama
# .pte + tokenizer, the four trained PCB models and their ONNX graphs, and the VisA test data
# for the benchmarks (+ 20 good pcb4 boards for the "teach" demo, + the ResNet-18 weights so the
# device needs no download at run time). About 1.1 GB.
#   bash scripts/make_bundle.sh      # -> outputs/qc_assist_bundle.tar
set -euo pipefail
cd "$(dirname "$0")/.."

files=(models/llama3_2_1b_8da4w.pte models/llama3_2_1b/tokenizer.model)
for c in pcb1 pcb2 pcb3 pcb4; do
  files+=("models/$c.pt" "models/${c}_embedder.onnx" "data/visa/$c/test" "data/visa/$c/ground_truth"
          "data/visa/$c/defect_labels.csv")
done
files+=(models/hailo/pcb*_embedder.hef)              # Hailo-10H NPU (AI HAT+ 2), ~16 MB
files+=(.cache/torch/hub/checkpoints/resnet18-*.pth)   # ImageNet backbone weights: no download on the device
# 20 evenly spaced good pcb4 boards for scripts/demo.py teach
mapfile -t teach < <(ls data/visa/pcb4/train/good/* | awk 'NR % 45 == 1' | head -20)
files+=("${teach[@]}")

mkdir -p outputs
tar -chf outputs/qc_assist_bundle.tar "${files[@]}"   # -h: store the tokenizer itself, not its symlink
ls -la outputs/qc_assist_bundle.tar
