# QC-Assist: On-device PCB Inspection with Llama Advisory on Arm

**Bharat AI-SoC Challenge 2026–27 — Phase 1 (Open Qualifying) technical report**

Problem Statement 5: Real-time industrial defect detection & quality-control assistant ·
Theme: Manufacturing & Industry 4.0

**Team:** [TEAM NAME] · **Institute:** [COLLEGE] · **Members:** [NAMES] · **Faculty mentor:** [MENTOR NAME]

**Code:** https://github.com/silentaf/BHARAT_SOC

> **Phase 1 scope.** The solution is implemented end to end and measured on an x86-64 laptop
> CPU and on the Arm CPU of an Android phone (§6.2, PS5 allows "Arm-based Android devices with
> camera"); the Hailo HEF is compiled and its INT8 accuracy verified in the emulator (§6.4).
> Raspberry Pi 5 + AI HAT+ 2 measurements are marked "Phase 2" below. Every number in this
> report comes from a JSON file in `results/` produced by the scripts in the repository.

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
- Defect type: for each defective board, the mean *residual* (patch feature minus its
  nearest normal patch) over the 8 most anomalous patches describes what changed; a
  logistic regression on it names the type (VisA PCB: melt, missing, bent, scratch; pcb4:
  missing, extra, dirt, damage). It reuses the detector's features, so it needs no extra
  network, works with every backend (CPU, ONNX, Hailo) and adds well under 1 ms. Below 50%
  confidence the board is reported as "unclassified" rather than guessed. The type model
  is stored in the product's model file; a factory trains it from labelled defect photos
  (`scripts/train_types.py`).

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
- Output checks: a generation is rejected if it contains a number (exactly as written) or a
  board location or defect type that is not in the facts, pairs a location or type with
  the wrong count, contradicts the trend, the critical count or the shift rating, claims
  no defect was found, repeats itself, or is cut off. Rejected output is replaced by
  deterministic text labelled `template`, and the dashboard shows the reason. Questions
  about people (the log has none) are answered by rule, and Hindi questions are answered
  with the exact Hindi figures from the log (see §7).
- Runner robustness: ExecuTorch 1.5.1's Python runner passes each token to Python as UTF-8
  text; when Llama 3's byte-level tokenizer splits a Devanagari character across two tokens
  the binding raises and generation stops. This is caught and treated as a failed
  generation (deterministic text shown). Giving defect types by their English names in
  Hindi prompts avoided the split in the benchmark.

### 4.3 Product features
- Operator dashboard served locally (no internet, no external assets).
- Teach a new board: capture N good boards from the camera, retrain on-device, hot-swap
  the model. Measured on the laptop: 3.0 s for 20 boards, including setting the alarm
  threshold by 5-fold scoring of the 20 boards (§6.6); Pi 5: Phase 2.

![Operator dashboard: a VisA PCB with bent pins outlined, rated MAJOR, type "bent", with the plant rule and Llama advice](images/dashboard.png)

*Operator dashboard (laptop, Phase 1): heatmap and outline, severity, defect type, vision
latency, Llama advice from ExecuTorch, shift statistics, questions, teach a new board.*

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
| Llama 8da4w + 4-bit embeddings | 860 MB `.pte`, 1.42 GB peak RAM, 31 tokens/s decode on the laptop CPU; Pi 5: Phase 2 |
| "Teach" threshold from 5-fold scoring of all 20 boards instead of 2 held-out boards | false alarms with 20 boards: pcb1 12% → 5%, pcb3 19% → 4%, pcb4 30% → 5% (pcb2 0% → 0%); recall 74/79/100% → 49/58/89% |
| Facts as text + few-shot + output checks instead of a JSON prompt | Llama outputs passing all checks: 17 of 18 (laptop benchmark); before, 1B output was frequently wrong (see §4.2) |
| Defect-type descriptor: residual vs nearest normal patch instead of the raw feature (pcb1, 1-NN, leave-one-out) | accuracy 0.79 → 0.89 |

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

### 6.2 Performance

Latency is per frame for the whole vision step (resize from the full camera image,
features, kNN, anomaly map), from `scripts/evaluate.py` on the 200 test boards.

| Platform / backend | ms per frame (mean / p95) | FPS |
|---|---|---|
| Laptop CPU (x86-64, 16 threads), PyTorch | 43.2 / 47.1 (mean of 4 products) | 23.4 |
| Laptop CPU (x86-64, 16 threads), ONNX Runtime | 40.2 / 46.5 (mean of 4 products) | 24.9 |
| Pi 5 CPU, PyTorch | Phase 2 | Phase 2 |
| Pi 5 CPU, ONNX Runtime | Phase 2 | Phase 2 |
| Pi 5 + AI HAT+ 2, HailoRT | Phase 2 | Phase 2 |

The laptop is shared and was busy (load average 14) during the final run; pcb1 with ONNX
Runtime measured 33.3 and 39.8 ms in two back-to-back runs. Earlier quiet-machine runs gave
37.3 / 32.8 ms (PyTorch / ONNX Runtime). The Pi numbers will be taken on an idle board.

| Llama 3.2 1B (ExecuTorch, 8da4w, XNNPACK) | Laptop (x86-64) | Pi 5 |
|---|---|---|
| Prefill tokens/s (median) | 244 | Phase 2 |
| Decode tokens/s (median) | 31.0 | Phase 2 |
| Defect advice, request to text (median) | 2.3 s | Phase 2 |
| Load + first call | 6.5 s | Phase 2 |
| Peak RAM (whole process) | 1,422 MB | Phase 2 |

Source: `scripts/bench_llama.py --repeats 3` → `results/x86_64/llama_VLSI.json`, which also
holds every generated text and whether it passed the checks:

| Task | Llama output accepted |
|---|---|
| Advice EN (major, critical) | 6/6 |
| Advice HI | 3/3 |
| Shift report EN / HI | 3/3, 2/3 (one "3.0%" for a 2.9% rate, caught) |
| Question EN | 3/3 |
| Question HI | not sent to Llama: Hindi figures from the log (Llama failed in earlier runs) |
| Question about people | not sent to Llama: answered by rule |

CPU contention: with vision and Llama both on the laptop CPU, vision time per frame rose
from 33 ms to ~100 ms while Llama was generating (live dashboard). This is the case for
running the CNN on the Hailo NPU.

**On Arm.** The same code and models on the phone's Arm CPU (Ubuntu in Termux, PyPI aarch64
wheels; `scripts/arm_benchmarks.sh`, setup in `docs/phone.md`):

<!-- arm-results:begin -->
**Arm device (Android phone, Arm CPU):** measurements pending — `bash scripts/arm_benchmarks.sh` on the phone ([`docs/phone.md`](docs/phone.md)), then `python scripts/arm_results.py`.
<!-- arm-results:end -->

### 6.3 Defect types

Cross-validated (5 x stratified 5-fold) on the 100 labelled defective test boards of each
VisA PCB category; `scripts/train_types.py` → `results/x86_64/<board>_types_VLSI.json`.
"Baseline" always answers the most common type.

| Category | Types (examples) | Accuracy | Balanced accuracy | Baseline | Confident (≥ 50%): share / accuracy |
|---|---|---|---|---|---|
| pcb1 | melt 54, missing 17, scratch 15, bent 14 | 88% | 84% | 54% | 95% / 90% |
| pcb2 | melt 55, missing 17, scratch 16, bent 12 | 76% | 68% | 55% | 85% / 82% |
| pcb3 | melt 43, scratch 22, bent 19, missing 16 | 82% | 79% | 43% | 84% / 88% |
| pcb4 | dirt 42, missing 26, extra 18, damage 8 | 69% | 58% | 45% | 82% / 75% |

Boards with several defect types count as correct if the predicted type is any of them.

### 6.4 Hailo NPU (INT8)

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

NPU latency and power: Phase 2, on the AI HAT+ 2.

Power (USB-C meter on the Pi 5: idle, inspecting, inspecting + Llama): Phase 2.

### 6.5 Arm deployment readiness

| Piece | State in Phase 1 |
|---|---|
| Llama 3.2 1B `.pte` (ExecuTorch, XNNPACK, 8da4w) | Exported and run through the same `TextLLMRunner` API that runs on the Pi; XNNPACK uses Arm KleidiAI kernels on Cortex-A76 |
| Vision HEF for Hailo-10H (AI HAT+ 2) | Compiled with the Hailo DFC 5.4 for all four boards; INT8 accuracy verified in the DFC emulator (§6.4) |
| HailoRT backend (`--backend hailo`) | Implemented with HailoRT's `VDevice` / `create_infer_model` API; runs on the device in Phase 2 |
| Python dependencies | ExecuTorch 1.5.1, PyTorch 2.14.1 and ONNX Runtime publish aarch64 wheels: `pip install -r requirements.txt` on Raspberry Pi OS 64-bit |
| Arm benchmarks | One command: `bash scripts/arm_benchmarks.sh` (phone in Phase 1, Pi 5 in Phase 2) |

### 6.6 Learning from few good boards

Image AUROC on the full test split when the model is taught from only N good boards
(`results/x86_64/teach_curve_VLSI.json`; fitting takes 0.3 s / 0.6 s / 1.4 s on the laptop):

| Good boards | pcb1 | pcb2 | pcb3 | pcb4 | mean |
|---|---|---|---|---|---|
| 10 | 0.862 | 0.856 | 0.874 | 0.949 | 0.885 |
| 20 | 0.911 | 0.906 | 0.884 | 0.980 | 0.920 |
| 50 | 0.935 | 0.925 | 0.919 | 0.993 | 0.943 |
| 200 (§6.1) | 0.972 | 0.932 | 0.924 | 0.990 | 0.955 |

With 20 boards the alarm threshold is set by 5-fold scoring (§4.3), which keeps false
alarms near 5%; detection of subtle defects is lower than with 200 boards (pcb2 recall
stays low at 20 boards), so a factory starts with ~20 and adds boards as they pass.

## 7. Limitations

- VisA images are captured under controlled lighting; real-line robustness needs our own
  captures (Phase 2).
- Defect types need labelled examples per product, and accuracy is lower for rare types
  (pcb4: balanced accuracy 58%; three types with only 2 examples each were left out). The
  dashboard demo on VisA test images uses a type model trained on those same images; the
  accuracy figures in §6.3 are cross-validated.
- Llama 3.2 1B is weak in Hindi question answering: in our benchmark runs its Hindi answers
  named wrong locations or called every defect "melt", so Hindi questions get the exact
  log figures in Hindi instead. Hindi advice and shift reports from Llama pass the checks.
- The checks catch wrong numbers, locations and trend/severity contradictions, but not
  every wording error (e.g. "lowest" next to the largest count with correct numbers).
- Severity cut-offs are heuristics and must be tuned per product with a quality engineer.

## 8. Demo

The Phase 1 demo video follows `docs/demo_script.md`: live inspection with defect types
and Llama advice, operator questions and the shift report, Hindi, and teaching a new board
type from 20 good boards. It runs on the laptop with `scripts/demo.py`.

## 9. Phase 2 plan

- Run every benchmark on the Raspberry Pi 5 (CPU) and Pi 5 + AI HAT+ 2 (HailoRT): vision
  latency, Llama tokens/s, RAM and power. Code, `.pte` and HEF are ready (§6.5).
- Our own boards under a fixed camera with controlled lighting; teach them on the device.
- Operator labelling of defect types from the dashboard, so the type model grows on the line.
- Hindi voice output for operators.

## 10. Reproducibility

All code, configs and scripts: https://github.com/silentaf/BHARAT_SOC (datasets and model
weights are downloaded by the scripts, not redistributed). Every table above is produced by:

```
python -I scripts/prepare_visa.py --tar VisA_20220922.tar --csv 1cls.csv --out data/visa
bash scripts/benchmark_all.sh                      # train + accuracy/latency, torch and ONNX (§6.1, §6.2)
python scripts/train_types.py --model models/pcb1.pt --data data/visa/pcb1   # defect types (§6.3)
python scripts/export_llama.py && python scripts/bench_llama.py --repeats 3   # Llama (§6.2)
bash scripts/compile_hailo.sh                      # HEF + INT8 emulator accuracy (§6.4)
```

Licenses: VisA CC BY 4.0 (Zou et al., ECCV 2022); Llama 3.2 Community License.
