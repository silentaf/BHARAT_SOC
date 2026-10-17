#!/usr/bin/env bash
# Set up QC-Assist on an Arm (aarch64) Linux device: a Raspberry Pi 5 (Raspberry Pi OS 64-bit)
# or an Android phone running Ubuntu inside Termux (docs/phone.md). Run from the repo folder:
#   bash scripts/setup_arm.sh [path/to/qc_assist_bundle.tar]
# The bundle (models + VisA test data) is made on the laptop with scripts/make_bundle.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(uname -m)" = aarch64 ] || echo "warning: this is $(uname -m), not aarch64"

# System packages: Python venv, and the libraries OpenCV needs.
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -q
$SUDO apt-get install -y -q python3 python3-venv python3-pip libgl1 libglib2.0-0 curl

python3 -m venv .venv
# PyPI has aarch64 wheels for everything (torch 2.14.1, executorch 1.5.1, onnxruntime): nothing compiles.
.venv/bin/pip install --no-cache-dir -q -r requirements.txt

BUNDLE=${1:-qc_assist_bundle.tar}
if [ -f "$BUNDLE" ]; then
  tar -xf "$BUNDLE"
  echo "unpacked $BUNDLE"
else
  echo "no bundle at $BUNDLE: copy it from the laptop (scripts/make_bundle.sh) and run: tar -xf <bundle>"
fi

.venv/bin/pytest -q
echo
echo "Setup done. Next:"
echo "  bash scripts/arm_benchmarks.sh        # Arm numbers for the report (~15 min)"
echo "  .venv/bin/python -m qc_assist          # start the app, then open http://127.0.0.1:8000"
