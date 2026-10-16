# QC-Assist: On-device PCB Inspection with Llama Advisory on Arm

**Bharat AI-SoC Challenge 2026–27 — Phase 1 (Open Qualifying) technical report**
Problem Statement 5: Real-time industrial defect detection & quality-control assistant
Team: _<team name>_ · _<college>_ · Faculty mentor: _<name>_

> Draft. Every `TBD` is filled only from measured results (`outputs/benchmarks/*.json`,
> the inspection log). Do not round up or estimate numbers in this report.

## 1. Problem

Electronics assembly lines inspect every printed circuit board for defects such as
missing or misplaced components, solder bridges, scratches and contamination. Manual
visual inspection is slow and inconsistent, and cloud-based AI inspection is often not an
option: factory networks are frequently isolated, images of products are confidential,
and a round trip to the cloud adds latency on a moving line.

QC-Assist performs the whole job on a Raspberry Pi 5: it detects and localises defects in
real time, rates their severity, logs every board, and uses Llama 3.2 1B running locally
through ExecuTorch to tell the operator what to do — in English or Hindi.

## 2. Design choices

**Good-only anomaly detection.** Real lines produce few defective boards and many defect
types appear only rarely, so collecting and labelling a defect dataset is the main
barrier to deployment. QC-Assist uses a PatchCore-style method that learns only from
defect-free boards: a pretrained CNN describes every image patch, a compact
memory bank of normal patches is kept (greedy coreset subsampling), and a patch is
anomalous if it is far from every normal patch. A new product is onboarded by showing
it ~20 good boards ("Teach a new board", §4.3).

**Separate pipelines on one SoC.** The CNN runs on the Hailo NPU (or the Arm CPU without
the HAT), the nearest-neighbour search and post-processing run on the Arm CPU, and Llama
runs on the Arm CPU through ExecuTorch with XNNPACK. Inspection never waits for the
LLM: language tasks run on a single background worker.

**Numbers from code, words from Llama.** All statistics (counts, rates, locations) are
computed deterministically from the vision output and the SQLite inspection log. Llama
receives them as facts and only explains them, so a 1B-parameter model is never asked
to compute or recall numbers.

## 3. System architecture

```
Vision pipeline                                    Language pipeline (ExecuTorch, Arm CPU)
camera -> resize -> patch-feature CNN (NPU/CPU)    verdict + log statistics
       -> kNN vs memory bank (CPU)                     -> Llama 3.2 1B, 4-bit (8da4w), XNNPACK
       -> anomaly map -> severity, area, location      -> advice / shift report / answers
       -> SQLite log -> dashboard                      -> dashboard
```

Hardware: Raspberry Pi 5 (8 GB, 4x Cortex-A76), Raspberry Pi AI HAT+ (Hailo-8/8L),
Camera Module 3.

## 4. Implementation

### 4.1 Vision
- Backbone: ResNet-18 (ImageNet weights), features from `layer2` + `layer3`,
  3x3 local average pooling, concatenated to 384-d patch features on a 40x40 grid (320x320 input).
- Memory bank: 200 evenly spaced good boards, greedy k-center coreset keeping 1% of patches
  (3,200 x 384 per product, 4.9 MB).
- Threshold: 99th percentile of anomaly scores on 200 *other* good boards; scores are
  normalised so 1.0 is the decision threshold.
- Severity: from score and anomalous area (minor / major / critical); location reported
  as a 3x3 board region.

### 4.2 Llama on ExecuTorch
- Model: Llama 3.2 1B Instruct, exported with ExecuTorch, 4-bit weights / 8-bit dynamic
  activations (8da4w), KV cache, XNNPACK backend. `.pte` size: TBD.
- Tasks: defect advice, end-of-shift report, operator questions over log statistics.
- Languages: English, Hindi.

### 4.3 Product features
- Operator dashboard served locally (no internet, no external assets).
- Teach a new board: capture N good boards from the camera, retrain on-device, hot-swap
  the model. Measured teach time on Pi 5: TBD.

## 5. Optimisation

| Step | Effect (measured) |
|---|---|
| Input 224 → 320 px (pcb1) | image AUROC 0.900 → 0.972, recall 50% → 80%; 21 → 38 ms per frame |
| Threshold calibrated on 200 separate good boards instead of a 20-board hold-out (pcb1, 224 px) | false alarms 14% → 2% |
| Coreset distance update as one mat-vec instead of `cdist` | 82.6 → 4.8 ms per iteration; training 277 s → 32.5 s (pcb1, 320 px), identical accuracy |
| 10% instead of 1% coreset | training 723 s (pcb1, 224 px) and 10x larger kNN — rejected for the Pi |
| ONNX Runtime instead of PyTorch eager (laptop CPU, pcb1) | 37.6 → 32.9 ms per frame |
| Disable ONNX Runtime thread spin-waiting | 166 → 11 ms per frame (ORT threads were starving the PyTorch kNN step) |
| Hailo NPU INT8 | TBD |
| Llama 8da4w quantisation + KleidiAI | TBD tokens/s |

## 6. Results

### 6.1 Detection accuracy (VisA PCB, official one-class split)

| Category | Image AUROC | Pixel AUROC | Recall @ threshold | False-alarm rate |
|---|---|---|---|---|
| pcb1 | 0.972 | 0.996 | 80% | 5% |
| pcb2 | 0.932 | 0.986 | 81% | 5% |
| pcb3 | 0.924 | 0.989 | 53% | 3% |
| pcb4 | 0.990 | 0.985 | 91% | 3% |
| **mean** | **0.955** | **0.989** | **76%** | **4%** |

Operating point: 99th-percentile threshold (favours few false alarms). pcb3's low recall at
this threshold despite AUROC 0.924 means a lower threshold would catch more defects at the
cost of more false alarms; the threshold percentile is a per-product setting.

#### 6.2 Performance

Latency is per frame for the whole vision step (resize from the full camera image,
features, kNN, anomaly map), from `scripts/evaluate.py` on the 200 test boards.

| Platform / backend | ms per frame (mean / p95) | FPS |
|---|---|---|
| Laptop CPU (x86-64, 16 threads), PyTorch | 37.3 / 38.7 (mean of 4 products) | 26.9 |
| Laptop CPU (x86-64, 16 threads), ONNX Runtime | 32.8 / 35.2 (mean of 4 products) | 30.5 |
| Pi 5 CPU, PyTorch | TBD | TBD |
| Pi 5 CPU, ONNX Runtime | TBD | TBD |
| Pi 5 + AI HAT+, HailoRT | TBD | TBD |

| Llama 3.2 1B (ExecuTorch) | Laptop | Pi 5 |
|---|---|---|
| Prefill tokens/s | TBD | TBD |
| Decode tokens/s | TBD | TBD |
| Time to first advice after a defect | TBD | TBD |
| Peak RAM | TBD | TBD |

Power (USB-C meter, Pi 5): idle TBD W, inspecting TBD W, inspecting + Llama TBD W.

## 7. Limitations

- VisA images are captured under controlled lighting; real-line robustness needs our own
  captures (Phase 2).
- Defect *type* is not classified yet; Llama is told location, size and severity only.
- Severity cut-offs are heuristics and must be tuned per product with a quality engineer.

## 8. Phase 2 plan

- HailoRT backend on the AI HAT+, INT8 accuracy check.
- Defect-type classifier on detected regions.
- Hindi voice output for operators.
- Field test with real boards on a demo conveyor or turntable.

## 9. Reproducibility

All code, configs and scripts: https://github.com/silentaf/BHARAT_SOC. Datasets and weights are downloaded by
the scripts and not redistributed. Licenses: VisA CC BY 4.0; Llama 3.2 Community License.
