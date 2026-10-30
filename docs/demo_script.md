# Phase 1 demo video: script and shot list

Target length: 4–5 minutes, 1080p screen recording with voice-over.
Recorder: OBS Studio, or GNOME's built-in recorder (Ctrl+Shift+Alt+R starts and stops it).

## Before recording

```bash
cd ~/"Desktop/bharat soc"                 # the project folder
.venv/bin/python scripts/demo.py inspect  # leave running; open http://127.0.0.1:8000
```

- Wait ~30 s after starting: the first Llama call loads the model (~6.5 s on the laptop).
- Turn **Wi-Fi off** before scene 3 (the dashboard shows "offline"; everything runs locally).
- Browser zoom ~110%, close other tabs, hide bookmarks bar.
- Have these open in tabs/windows for scene 7: `results/x86_64/summary.md`,
  `docs/hailo.md` (section 3), `results/x86_64/llama_VLSI.json`, a terminal.

## Recorded clips

`python scripts/record_demo.py` records the laptop scenes (1–7 and the README for 8) as silent
1080p clips in `outputs/demo_video/` (Playwright driving Chrome; about 10 minutes). Add the
voice-over below in a video editor. The Ask scene is retaken (max 3) until Llama's own answer
passes the checks; when a check replaces a Llama answer with the exact figures, the dashboard
says so, and the voice-over can point at it.

## Phone clip (scene 8, record it first, ~10 min)

1. Phone plugged in, **mobile data and Wi-Fi off** (shows it is offline).
2. Termux: `proot-distro login ubuntu`, then `cd BHARAT_SOC && .venv/bin/python scripts/demo.py inspect`.
   Wait until it prints `Running on http://127.0.0.1:8000` (about 2 minutes on the phone).
3. Swipe Termux away to the background (don't close it), open **Chrome** → `127.0.0.1:8000`.
4. Start the phone's screen recorder (Xiaomi: Control centre → **Screen recorder**), record
   ~60 s: boards passing, a red defect outline, the Llama advice card filling in (~7–10 s).
5. Copy the recording to the laptop (USB) and cut 20 s of it into scene 8.

## Scenes

| # | Time | Show | Say (suggested) |
|---|---|---|---|
| 1 | 0:00–0:15 | README top (title + screenshot) | "This is QC-Assist, our entry for Problem Statement 5: real-time defect detection and a quality-control assistant, running fully on the device." |
| 2 | 0:15–0:40 | README "Architecture" diagram | "Electronics lines inspect every circuit board by eye. Cloud AI is often not allowed: factory networks are offline and product images are confidential. QC-Assist does everything on an Arm device. Two separate pipelines: the vision network, which on a Raspberry Pi 5 runs on the Hailo NPU, and Llama 3.2 on the Arm CPU through ExecuTorch. In this video both run on CPUs: the laptop, then an Arm phone." |
| 3 | 0:40–1:50 | Dashboard, `inspect` scene. Point at the heatmap, the verdict line (severity, **type**, score, ms), then an advisory card | "Boards stream past. Good boards pass. Here a defect: the heatmap and red outline show where, it is rated MAJOR, and its type is named — bent pins. The model learned what a good board looks like from defect-free boards only. The first line of the advice is the plant rule — remove the board, keep the line running — decided by code, never by the AI. Below it, Llama 3.2 1B, running through ExecuTorch on this CPU, explains what to look for and the usual cause. The label shows it came from llama-executorch." |
| 4 | 1:50–2:30 | Type "Where are most defects this shift?" → Ask; then **Shift report** | "Operators can ask about the shift. Numbers come from the inspection log; Llama only phrases them, and every answer is checked against the facts before it is shown — wrong numbers or locations are replaced by the exact figures. Here is the speed: over 20 tokens per second on this laptop. And the end-of-shift report for the supervisor." |
| 5 | 2:30–2:55 | Click **English → हिंदी**; wait for a new advisory; click Shift report | "For operators who prefer Hindi, the same advice and report in Hindi." |
| 6 | 2:55–3:45 | Terminal: Ctrl+C, then `.venv/bin/python scripts/demo.py teach`. Dashboard shows NO MODEL. Paste the printed folder into "Train from folder", click it | "A new product. No model yet. We give it 20 good boards of the new design… trained in about three seconds, and it is already inspecting the new board. No defect images, no retraining in the cloud." |
| 7 | 3:45–4:30 | `results/x86_64/summary.md`; `docs/hailo.md` table; terminal `ls -la models/hailo/*.hef` | "Everything is measured. On the VisA PCB dataset, mean image AUROC 0.955; defect types 69 to 88 percent correct. We compiled the vision network for the Hailo-10H NPU with the Dataflow Compiler — in INT8 the accuracy is unchanged. Llama is exported with ExecuTorch, 4-bit, 860 megabytes." |
| 8 | 4:30–4:50 | **Phone screen recording** (see "Phone clip" below): the dashboard in the phone's Chrome, a defect, the Llama advice appearing | "The same app on an Android phone's Arm CPU, no laptop, no internet: vision at 3 frames per second with the same accuracy, Llama at 12 tokens per second through ExecuTorch. In Phase 2 the Raspberry Pi 5 with the AI HAT+ 2 moves the vision network onto the Hailo NPU." |
| 9 | 4:50–5:00 | Dashboard | "QC-Assist: offline, on-device inspection that learns a new board from 20 samples. Thank you." |

## Honesty notes (if asked)

- The `inspect` sequence is curated (good boards the detector passes, defects it catches) so
  the video shows each feature; full-test-set accuracy, including misses and false alarms,
  is in the report.
- The demo's defect-type model was trained on these VisA defect images; its accuracy on
  unseen defects is the cross-validated figure (69–88%).
- Phase 1 runs on an x86 laptop and on an Android phone (Arm CPU, results/aarch64); Pi 5 + AI HAT+ 2 is Phase 2 (docs/pi.md).
