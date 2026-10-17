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

**Code decides, Llama explains, code checks.** The required action and whether to stop the
line come from a fixed plant policy (English and Hindi), never from the model. All
statistics (counts, rates, locations, trend, a GOOD/POOR shift rating) are computed
deterministically from the vision output and the SQLite inspection log; Llama receives
them as plain-language facts and only explains them. Every generated text is checked
before it is shown (§4.2). We arrived at this after measuring what Llama 3.2 1B does with
plain prompts: it told the operator to stop the line for a major defect and called a
rising defect rate "decreasing".

## 3. System architecture

```
Vision pipeline                                    Language pipeline (ExecuTorch, Arm CPU)
camera -> resize -> patch-feature CNN (NPU/CPU)    verdict + log statistics
       -> kNN vs memory bank (CPU)                     -> Llama 3.2 1B, 4-bit (8da4w), XNNPACK
       -> anomaly map -> severity, area, location      -> advice / shift report / answers
       -> SQLite log -> dashboard                      -> dashboard
```

Hardware: Raspberry Pi 5 (8 GB, 4x Cortex-A76), Raspberry Pi AI HAT+ 2 (Hailo-10H; the
vision network is compiled for it with DFC 5.4, and the same flow targets Hailo-8/8L with
DFC 3.x), Camera Module 3.

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
- Model: Llama 3.2 1B Instruct, exported with ExecuTorch 1.5.1: 4-bit grouped weights with
  8-bit dynamic activations (8da4w, group 32), 4-bit embeddings, KV cache with fused SDPA,
  XNNPACK backend, 768-token context. `.pte` size: 860 MB. Runs in-process through
  `TextLLMRunner` (loaded once, kept in memory) on a single background worker.
- Tasks: defect advice ("what to look for, usual cause"), end-of-shift report, operator
  questions over the log statistics. Languages: English, Hindi.
- Prompting: worked examples (few-shot) per task; the shift-report example is chosen to
  match the shift (good or poor). Facts are given in English, with Hindi location names
  for Hindi replies (Devanagari costs ~3x the tokens and the model reads English facts
  more reliably).
- Output checks: a generation is rejected if it contains a number or a board location
  that is not in the facts, contradicts the trend, the critical count or the shift rating,
  claims no defect was found, repeats itself, or is cut off. Rejected output is replaced by
  deterministic text labelled `template`, and the dashboard shows the reason. Questions
  about people (the log has none) are answered by rule.

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
| ONNX Runtime instead of PyTorch eager (laptop CPU, pcb1, quiet machine) | 37.6 → 32.9 ms per frame |
| Disable ONNX Runtime thread spin-waiting | 166 → 11 ms per frame (ORT threads were starving the PyTorch kNN step) |
| ONNX graph with one output per backbone layer (upsample + concat moved to CPU) | Hailo quantisation failed with one concatenated output (layers need separate scales); compiles with two |
| Hailo NPU INT8 (DFC emulator, 4 boards) | mean image AUROC 0.955 → 0.956, pixel AUROC 0.989 → 0.989: no measurable loss |
| Llama 8da4w + 4-bit embeddings | 860 MB `.pte`, 1.47 GB peak RAM, 31 tokens/s decode on the laptop CPU; Pi 5 TBD |
| Facts as text + few-shot + output checks instead of a JSON prompt | Llama outputs passing all checks: 17 of 21 (laptop benchmark); before, 1B output was frequently wrong (see §4.2) |

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
| Laptop CPU (x86-64, 16 threads), PyTorch | 43.2 / 47.1 (mean of 4 products) | 23.4 |
| Laptop CPU (x86-64, 16 threads), ONNX Runtime | 40.2 / 46.5 (mean of 4 products) | 24.9 |

The laptop is shared and was busy (load average 14) during the final run; pcb1 with ONNX
Runtime measured 33.3 and 39.8 ms in two back-to-back runs. Earlier quiet-machine runs gave
37.3 / 32.8 ms (PyTorch / ONNX Runtime). The Pi numbers will be taken on an idle board.
| Pi 5 CPU, PyTorch | TBD | TBD |
| Pi 5 CPU, ONNX Runtime | TBD | TBD |
| Pi 5 + AI HAT+, HailoRT | TBD | TBD |

| Llama 3.2 1B (ExecuTorch, 8da4w, XNNPACK) | Laptop (x86-64) | Pi 5 |
|---|---|---|
| Prefill tokens/s (median) | 240 | TBD |
| Decode tokens/s (median) | 31.4 | TBD |
| Defect advice, request to text (median, English) | 2.1 s | TBD |
| Load + first call | 6.5 s | TBD |
| Peak RAM (whole process) | 1,469 MB | TBD |

Source: `scripts/bench_llama.py --repeats 3` → `results/x86_64/llama_VLSI.json`, which also
holds every generated text and whether it passed the checks:

| Task | Llama output accepted |
|---|---|
| Advice EN (major, critical) | 6/6 |
| Advice HI | 2/3 |
| Shift report EN / HI | 3/3, 3/3 |
| Question EN | 3/3 |
| Question HI | 0/3, all caught (wrong location); correct Hindi figures shown instead |

CPU contention: with vision and Llama both on the laptop CPU, vision time per frame rose
from 33 ms to ~100 ms while Llama was generating (live dashboard). This is the case for
running the CNN on the Hailo NPU.

### 6.3 Hailo NPU (INT8)

Compiled with the Hailo Dataflow Compiler 5.4.0 for Hailo-10H (AI HAT+ 2): HEFs 3.9–4.4 MB,
compile time 122 s per board on the laptop CPU. Accuracy of the INT8 model with the DFC emulator
(same memory bank and threshold as FP32):

| Category | Image AUROC FP32 → INT8 | Pixel AUROC FP32 → INT8 | Recall | False alarms |
|---|---|---|---|---|
| pcb1 | 0.972 → 0.972 | 0.996 → 0.996 | 80% → 85% | 5% → 5% |
| pcb2 | 0.932 → 0.931 | 0.986 → 0.986 | 81% → 83% | 5% → 5% |
| pcb3 | 0.924 → 0.931 | 0.989 → 0.989 | 53% → 55% | 3% → 3% |
| pcb4 | 0.990 → 0.990 | 0.985 → 0.984 | 91% → 92% | 3% → 3% |
| **mean** | **0.955 → 0.956** | **0.989 → 0.989** | **76% → 79%** | **4% → 4%** |

NPU latency and power: TBD on hardware.

Power (USB-C meter, Pi 5): idle TBD W, inspecting TBD W, inspecting + Llama TBD W.

## 7. Limitations

- VisA images are captured under controlled lighting; real-line robustness needs our own
  captures (Phase 2).
- Defect *type* is not classified yet; Llama is told location, size and severity only.
- Llama 3.2 1B is weak in Hindi: free-form Hindi answers to operator questions failed the
  checks in all 3 benchmark runs (the deterministic Hindi summary is shown instead).
  Hindi advice and reports pass.
- The checks catch wrong numbers, locations and trend/severity contradictions, but not
  every wording error (e.g. "lowest" next to the largest count with correct numbers).
- Severity cut-offs are heuristics and must be tuned per product with a quality engineer.

## 8. Phase 2 plan

- HailoRT backend measured on the AI HAT+ (code and HEF ready; INT8 accuracy already checked
  in the emulator).
- Defect-type classifier on detected regions.
- Hindi voice output for operators.
- Field test with real boards on a demo conveyor or turntable.

## 9. Reproducibility

All code, configs and scripts: https://github.com/silentaf/BHARAT_SOC. Datasets and weights are downloaded by
the scripts and not redistributed. Licenses: VisA CC BY 4.0; Llama 3.2 Community License.
