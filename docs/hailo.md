# Vision on the Hailo NPU (AI HAT+)

PS5 rules: if Hailo is used, the vision model runs through **HailoRT on the Hailo NPU**,
while **Llama runs through ExecuTorch on the Arm CPU**: two separate pipelines.

```
Vision pipeline (HailoRT)                          Llama pipeline (ExecuTorch)
camera -> resize -> [NPU] ResNet-18 layer2+layer3   inspection log + verdict
                          + 3x3 local averaging        -> [CPU] Llama 3.2 1B (XNNPACK/KleidiAI)
       -> [CPU] upsample + concat, kNN vs memory bank  -> advice / report / answers
       -> [CPU] anomaly map, severity
```

Only the CNN goes to the NPU. The kNN search against the memory bank stays on the
Arm CPU (it is a small matrix multiply + min, not a good fit for the NPU).

Moving the CNN off the CPU matters for more than speed: on the laptop, vision went from
33 ms to ~100 ms per frame while Llama was generating, because both share the CPU.

## Which chip, which compiler

| Board | Chip | `--hw-arch` | Dataflow Compiler |
|---|---|---|---|
| AI HAT+ 13 TOPS | Hailo-8L | `hailo8l` | **3.x** |
| AI HAT+ 26 TOPS | Hailo-8 | `hailo8` | **3.x** |
| AI HAT+ 2 | Hailo-10H | `hailo10h` | **5.x** (verified: 5.4.0) |

DFC 5.4 lists Hailo-8/8L as deprecated and only accepts `hailo10h`, `hailo10p` and
`hailo15*`. Download the DFC that matches the HAT you will use; the steps are the same.

## 1. Install the DFC (x86 Linux, own venv)

The DFC pins TensorFlow 2.19, numpy 1.26 and onnxsim 0.4.36 (no wheel for Python ≥ 3.12),
so it gets its own Python 3.10 venv, kept inside the project:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv/python" UV_PYTHON_PREFERENCE=only-managed
.venv/bin/pip install uv
.venv/bin/uv python install 3.10
.venv/bin/uv venv --python 3.10 .venv-hailo
WHL=~/Downloads/hailo_dataflow_compiler-5.4.0-py3-none-linux_x86_64.whl
# every dependency except pygraphviz (needs system graphviz headers; only used for drawing graphs)
unzip -p $WHL '*.dist-info/METADATA' | sed -n 's/^Requires-Dist: //p' | grep -v '^pygraphviz' > /tmp/dfc_reqs.txt
.venv/bin/uv pip install --python .venv-hailo/bin/python -r /tmp/dfc_reqs.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/uv pip install --python .venv-hailo/bin/python --no-deps $WHL
.venv/bin/uv pip install --python .venv-hailo/bin/python numpy==1.26.4 "opencv-python-headless<4.11" scikit-learn pyyaml
.venv-hailo/bin/hailo --version     # Hailo Dataflow Compiler v5.4.0
```

## 2. Export, compile, check INT8 accuracy

```bash
.venv/bin/python scripts/export_onnx.py --model models/pcb1.pt --calib data/visa/pcb1/train/good
bash scripts/compile_hailo.sh                      # all boards; HW_ARCH=hailo10h by default
```

`export_onnx.py` writes `models/<name>_embedder.onnx` (raw 0..255 RGB in, normalisation inside
the graph, checked against PyTorch to ~2e-6) and a 64-image calibration set. The graph has
**one output per backbone layer** (layer2 40x40x128, layer3 20x20x256); upsampling and concat
run on the CPU. A single concatenated output failed in `hailo optimize`
(`AccelerasNumerizationError: the vector must be a scalar`): the two layers need separate
quantisation scales.

`compile_hailo.sh` runs `hailo parser onnx` → `hailo optimize` (INT8, calibration set) →
`hailo compiler` and writes `models/hailo/<name>_embedder.hef`, then measures accuracy with
the DFC's INT8 emulator (`evaluate.py --backend hailo-emu`). It hides the GPU from the DFC
(`CUDA_VISIBLE_DEVICES=""`), so the DFC uses optimisation level 0 (calibration only); level
2 needs a GPU or ≥ 1024 calibration images.

Measured (laptop, i7-14650HX, CPU only, pcb1): optimize 20 s, compile 122 s. HEFs: 3.9–4.4 MB.

## 3. Results: INT8 Hailo model (emulated) vs FP32

`results/x86_64/*_hailo-emu_VLSI.json` vs `*_torch_VLSI.json`. Same memory bank and threshold
as FP32: only the CNN is quantised.

| Board | Image AUROC FP32 → INT8 | Pixel AUROC FP32 → INT8 | Recall | False alarms |
|---|---|---|---|---|
| pcb1 | 0.972 → 0.972 | 0.996 → 0.996 | 80% → 85% | 5% → 5% |
| pcb2 | 0.932 → 0.931 | 0.986 → 0.986 | 81% → 83% | 5% → 5% |
| pcb3 | 0.924 → 0.931 | 0.989 → 0.989 | 53% → 55% | 3% → 3% |
| pcb4 | 0.990 → 0.990 | 0.985 → 0.984 | 91% → 92% | 3% → 3% |

The emulator's latency (~1.8 s per frame) is x86 simulation, not NPU speed. NPU latency is
measured on the Pi.

## 4. Run on the Pi

```bash
sudo apt install hailo-all        # HailoRT + Python bindings (Raspberry Pi OS); AI HAT+ 2 may need h10-hailort
.venv/bin/python scripts/evaluate.py --model models/pcb1.pt --data data/visa/pcb1 --backend hailo --threads 4
.venv/bin/python scripts/dashboard.py --source 0 --model models/pcb1.pt --backend hailo
```

`HailoEmbedder` (`qc_assist/anomaly.py`) uses HailoRT's `VDevice` / `create_infer_model`
API: UINT8 NHWC input, FLOAT32 outputs, then the same CPU merge as the other backends.
**Not run on hardware yet.** The Pi's Python venv needs access to the system HailoRT
bindings (`python3 -m venv --system-site-packages .venv`).

## Numbers for the report

| Backend | Where | Full frame ms | FPS | Image AUROC |
|---|---|---|---|---|
| PyTorch | laptop CPU (shared, busy) | 43.2 | 23.4 | 0.955 (mean of 4) |
| ONNX Runtime | laptop CPU (shared, busy) | 40.2 | 24.9 | 0.955 (mean of 4) |
| Hailo INT8 (emulator) | laptop | n/a | n/a | 0.956 (mean of 4) |
| PyTorch | Pi 5 CPU | | | |
| ONNX Runtime | Pi 5 CPU | | | |
| HailoRT (INT8) | AI HAT+ NPU | | | |
