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

## Scenes

| # | Time | Show | Say (suggested) |
|---|---|---|---|
| 1 | 0:00–0:15 | README top (title + screenshot) | "This is QC-Assist, our entry for Problem Statement 5: real-time defect detection and a quality-control assistant, running fully on the device." |
| 2 | 0:15–0:40 | README "Architecture" diagram | "Electronics lines inspect every circuit board by eye. Cloud AI is often not allowed: factory networks are offline and product images are confidential. QC-Assist does everything on an Arm device. Two separate pipelines: vision runs on the Hailo NPU through HailoRT, and Llama 3.2 runs on the Arm CPU through ExecuTorch." |
| 3 | 0:40–1:50 | Dashboard, `inspect` scene. Point at the heatmap, the verdict line (severity, **type**, score, ms), then an advisory card | "Boards stream past. Good boards pass. Here a defect: the heatmap and red outline show where, it is rated MAJOR, and its type is named — bent pins. The model learned what a good board looks like from defect-free boards only. The first line of the advice is the plant rule — remove the board, keep the line running — decided by code, never by the AI. Below it, Llama 3.2 1B, running through ExecuTorch on this CPU, explains what to look for and the usual cause. The label shows it came from llama-executorch." |
| 4 | 1:50–2:30 | Type "Where are most defects this shift?" → Ask; then **Shift report** | "Operators can ask about the shift. Numbers come from the inspection log; Llama only phrases them, and every answer is checked against the facts before it is shown — wrong numbers or locations are replaced by the exact figures. Here is the speed: about 31 tokens per second. And the end-of-shift report for the supervisor." |
| 5 | 2:30–2:55 | Click **English → हिंदी**; wait for a new advisory; click Shift report | "For operators who prefer Hindi, the same advice and report in Hindi." |
| 6 | 2:55–3:45 | Terminal: Ctrl+C, then `.venv/bin/python scripts/demo.py teach`. Dashboard shows NO MODEL. Paste the printed folder into "Train from folder", click it | "A new product. No model yet. We give it 20 good boards of the new design… trained in about three seconds, and it is already inspecting the new board. No defect images, no retraining in the cloud." |
| 7 | 3:45–4:30 | `results/x86_64/summary.md`; `docs/hailo.md` table; terminal `ls -la models/hailo/*.hef` | "Everything is measured. On the VisA PCB dataset, mean image AUROC 0.955; defect types 69 to 88 percent correct. We compiled the vision network for the Hailo-10H NPU with the Dataflow Compiler — in INT8 the accuracy is unchanged. Llama is exported with ExecuTorch, 4-bit, 860 megabytes." |
| 8 | 4:30–4:50 | README "Status" table | "In Phase 2, on the Raspberry Pi 5 with the AI HAT+ 2, we run the same code — the ExecuTorch and HailoRT pieces are ready — and measure speed and power on the device." |
| 9 | 4:50–5:00 | Dashboard | "QC-Assist: offline, on-device inspection that learns a new board from 20 samples. Thank you." |

## Honesty notes (if asked)

- The `inspect` sequence is curated (good boards the detector passes, defects it catches) so
  the video shows each feature; full-test-set accuracy, including misses and false alarms,
  is in the report.
- The demo's defect-type model was trained on these VisA defect images; its accuracy on
  unseen defects is the cross-validated figure (69–88%).
- Phase 1 runs on an x86 laptop; Arm/Pi numbers are Phase 2.
