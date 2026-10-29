# Phase 1 Plan — Bharat AI-SoC Challenge 2026–27, PS5

**Goal:** submit by **21 Nov 2026** (2-day buffer before the **23 Nov** deadline in our shortlisting email).
**Status (29 Oct):** shortlisted, PS5 confirmed. Laptop-side work done.
**Arm device for Phase 1: Android phone (POCO M6 Plus 5G) — DONE 30 Oct**: set up over SSH, benchmarks
in `results/aarch64/`, numbers in the report and README. Pi 5 + AI HAT+ 2: Phase 2 (`docs/pi.md`).
Left for the team:
4. Record the demo video (laptop dashboard + one scene of the app on the phone with its camera).
5. Fill `[TEAM NAME]` etc. in the report, add the video link to the README, fresh-clone test, submit.
**What to submit:** technical report + source code (GitHub) + demo video.

## What Phase 1 must prove (shortlisting bar)

The other early PS5 entries we found are still on laptops with rule-based advisors.
We aim to be ahead of them on exactly these four points:

1. **Runs on Arm hardware** — Raspberry Pi 5 (CPU-only is allowed; AI HAT+ is a bonus).
2. **Llama 3.2 1B actually running through ExecuTorch** on the Pi, not a placeholder.
3. **Real-time defect inspection** of PCBs with numbers: accuracy (AUROC), FPS, latency.
4. **Clean deliverables** — README anyone can follow, honest report, 3–5 min video.

Hailo NPU acceleration, Hindi voice, and polished UI are *nice to have* for Phase 1
and can move to Phase 2 (PoC, due 10 Feb 2027) if time runs short.

## Week by week

| Week | Dates | Work | Done when |
|---|---|---|---|
| 0 | 15–19 Oct | **Register (PS5) by 19 Oct.** Order Pi 5 8GB + AI HAT+ + Camera Module 3 + cooler + PSU. Request Llama 3.2 access on Hugging Face. Repo + venv. | Registration confirmation email; hardware ordered |
| 1 | 20–26 Oct | Download VisA (pcb1–pcb4). Train/evaluate anomaly model on laptop; compare backbones. Install ExecuTorch, export Llama 3.2 1B (4-bit), run it on laptop. | AUROC table for pcb1–4; Llama `.pte` generates text |
| 2 | 27 Oct–2 Nov | Pi setup: OS, camera, venv. Build ExecuTorch Llama runner on Pi. Run inspection on Pi CPU. **First full loop on Pi:** camera → defect → Llama advice → log. | End-to-end works on Pi; first FPS + tokens/s numbers |
| 3 | 3–9 Nov | Hailo: compile backbone to HEF (Hailo compiler runs on x86 laptop), run on NPU, kNN stays on CPU. Llama features: shift report + questions over the inspection log. | CPU vs NPU benchmark; Llama answers log questions |
| 4 | 10–14 Nov | Demo rig: camera stand, lighting, real PCBs (good + deliberately damaged). Full benchmarks on the Pi: FPS, latency, RAM, power (USB-C meter), Llama tokens/s. Attend the mentor doubt-clearing session. | Live inspection of real boards works; Pi numbers in report |
| 5 | 15–21 Nov | Record demo video. Fill report placeholders, polish README, final test from a fresh clone. **Submit via the Challenge Submission Form.** | Submitted |

## If something slips

- **AI HAT+ delayed** → Phase 1 runs Pi 5 CPU-only (allowed). Hailo moves to Phase 2.
- **Pi delayed** → keep building on laptop; Pi work compresses into weeks 3–4. Escalate early.
- **ExecuTorch build trouble on Pi** → follow Arm's Learning Path "Run Llama 3 on a Raspberry Pi 5 using ExecuTorch"; ask tech-queries@armbharatchallenge.com (subject: PS5).
- **Llama access not approved** → re-apply immediately; this blocks week 1, so request it on day 0.

## Deliverables checklist

**Source code (GitHub, public)**
- [x] README: what it does, architecture diagram, setup on laptop and Pi, how to reproduce numbers
- [x] Training, inspection, evaluation scripts
- [x] Llama export + run instructions (ExecuTorch)
- [x] Hailo pipeline documented separately from the Llama pipeline (required by PS5 rules)
- [x] No datasets or model weights committed; download instructions instead
- [x] Licenses noted (VisA: CC BY 4.0; MVTec AD: CC BY-NC-SA 4.0; Llama 3.2 Community License)

**Technical report**
- [x] Problem + why on-device (latency, factory networks are often offline/air-gapped, data stays on site)
- [x] System architecture: vision pipeline (NPU/CPU) and Llama pipeline (CPU, ExecuTorch)
- [x] Method: good-only anomaly detection, severity, advisory
- [ ] Optimisation: quantisation, threads, input size, NPU offload (laptop + Hailo emulator done; Pi numbers pending)
- [ ] Results: accuracy + performance tables, laptop vs Pi CPU vs Pi + Hailo (laptop done; Pi pending)
- [x] Limitations and Phase 2 plan

**Demo video (3–5 min)**
- [ ] Hardware shown on camera (Pi, HAT, camera rig)
- [ ] Good board passes; damaged board flagged with heatmap
- [ ] Llama advice appears; tokens/s visible
- [ ] Wi-Fi off to show it is fully offline
- [ ] Benchmark summary on screen
