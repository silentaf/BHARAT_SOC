# Vision on the Hailo NPU (AI HAT+)

PS5 rules: if Hailo is used, the vision model runs through **HailoRT on the Hailo NPU**,
while **Llama runs through ExecuTorch on the Arm CPU** — two separate pipelines.

```
Vision pipeline (HailoRT)                         Llama pipeline (ExecuTorch)
camera -> resize -> [NPU] patch-feature network   inspection log + verdict
       -> [CPU] nearest-normal search (kNN)          -> [CPU] Llama 3.2 1B (XNNPACK/KleidiAI)
       -> [CPU] anomaly map, severity                -> advice / report / answers
```

Only the CNN goes to the NPU. The kNN search against the memory bank stays on the
Arm CPU (it is a small matrix multiply + min, not a good fit for the NPU).

## Which chip

| Board | Chip | `--hw-arch` |
|---|---|---|
| AI HAT+ 13 TOPS | Hailo-8L | `hailo8l` |
| AI HAT+ 26 TOPS | Hailo-8 | `hailo8` |
| AI HAT+ 2 | Hailo-10H | uses a different compiler release — check Hailo docs |

## 1. Export (laptop, this repo)

```bash
.venv/bin/python scripts/export_onnx.py --model models/pcb1.pt --calib data/visa/pcb1/train/good
# -> models/pcb1_embedder.onnx, models/pcb1_calib.npy
```

The ONNX graph takes raw RGB pixels (0..255) — normalisation is inside the graph —
and is checked against PyTorch (max abs diff ~1e-6).

## 2. Compile to HEF (x86 Linux with the Hailo Dataflow Compiler)

Needs a free account on the Hailo Developer Zone to download the Dataflow Compiler (DFC).
Install it in its **own** venv (it pins its own Python/TensorFlow versions).
Indicative commands — check the DFC user guide for your version:

```bash
hailo parser onnx models/pcb1_embedder.onnx --hw-arch hailo8l
hailo optimize pcb1_embedder.har --calib-set-path models/pcb1_calib.npy --hw-arch hailo8l
hailo compiler pcb1_embedder_optimized.har --hw-arch hailo8l
# -> pcb1_embedder.hef
```

Quantisation to 8-bit happens in `optimize`; compare anomaly AUROC before/after
(`scripts/evaluate.py`) and report the drop, if any.

## 3. Run on the Pi

Raspberry Pi OS: `sudo apt install hailo-all` installs HailoRT and its Python bindings.
A `HailoEmbedder` (same interface as `OnnxEmbedder` in `qc_assist/anomaly.py`) will be
added when the hardware arrives, selected with `--backend hailo`.

## Numbers for the report

| Backend | Where | Backbone ms | Full frame ms | FPS | Image AUROC |
|---|---|---|---|---|---|
| PyTorch | laptop CPU | | | | |
| ONNX Runtime | laptop CPU | | | | |
| PyTorch | Pi 5 CPU | | | | |
| ONNX Runtime | Pi 5 CPU | | | | |
| HailoRT (INT8) | AI HAT+ NPU | | | | |
