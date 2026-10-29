#!/usr/bin/env bash
# All benchmarks for the report on an Arm device (phone or Raspberry Pi 5), one command:
#   DEVICE_NAME="Raspberry Pi 5" bash scripts/arm_benchmarks.sh   # THREADS=4 by default (Pi 5 has 4 cores)
# DEVICE_NAME is how the report names the device (e.g. the phone model); Termux hides it from Linux.
# Writes results/aarch64/: device info, vision accuracy/latency (PyTorch + ONNX Runtime) for
# pcb1-4, and the Llama benchmark. Copy that folder back to the laptop (or git push it).
set -euo pipefail
cd "$(dirname "$0")/.."
export THREADS=${THREADS:-4}
export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch"
OUT="results/$(uname -m)"
mkdir -p "$OUT"

{
  echo "device: ${DEVICE_NAME:-}"
  echo "host: $(hostname)   date: $(date -Iseconds)"
  grep -m1 -E "^(Model|Hardware)" /proc/cpuinfo || true
  lscpu | grep -E "^(Architecture|CPU\(s\)|Model name|Vendor ID|CPU max MHz)" || true
  grep MemTotal /proc/meminfo
  .venv/bin/python -c "import platform, torch, executorch; print('python', platform.python_version(), '| torch', torch.__version__, '| executorch', executorch.__version__ if hasattr(executorch, '__version__') else '?')"
  echo "threads for vision: $THREADS"
} | tee "$OUT/device_$(hostname).txt"

echo "== vision: 4 boards x (PyTorch, ONNX Runtime) =="
bash scripts/benchmark_all.sh

echo "== Llama 3.2 1B on ExecuTorch =="
.venv/bin/python scripts/bench_llama.py --repeats 3 | tail -25

echo "done: results in $OUT/ (on the laptop: python scripts/arm_results.py puts them in the report)"
