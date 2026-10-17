# Running QC-Assist on an Android phone (Arm)

PS5 allows "Arm-based Android devices with camera". This runs the **same code** on the phone's
Arm CPU: Ubuntu (aarch64) inside Termux, the PyPI aarch64 wheels of PyTorch, ExecuTorch and
ONNX Runtime, and the phone camera through the IP Webcam app. Vision and Llama both run on
the phone's Arm CPU (no Hailo NPU on a phone).

**Phone:** 64-bit Android, **6 GB RAM minimum (8 GB better)**, ~6 GB free storage.
Keep it plugged in while benchmarking.

## 1. On the laptop: make the bundle (models + test data, ~1.1 GB)

```bash
bash scripts/make_bundle.sh          # -> outputs/qc_assist_bundle.tar
```

Copy `outputs/qc_assist_bundle.tar` to the phone's **Download** folder (USB cable, "File
transfer" mode).

## 2. On the phone: install two apps

- **Termux** from F-Droid: https://f-droid.org/packages/com.termux/ (the Play Store version is outdated).
- **IP Webcam** (Pavel Khlebovich) from the Play Store. In its settings set video resolution to
  1280x720 and enable running in the background, then tap **Start server**.

## 3. In Termux: Ubuntu + QC-Assist

```bash
pkg update -y && pkg install -y proot-distro
termux-setup-storage                 # allow storage access
termux-wake-lock                     # stop Android from pausing Termux
proot-distro install ubuntu
proot-distro login ubuntu --bind /sdcard:/sdcard
```

Now inside Ubuntu:

```bash
apt update && apt install -y git
git clone https://github.com/silentaf/BHARAT_SOC.git && cd BHARAT_SOC
bash scripts/setup_arm.sh /sdcard/Download/qc_assist_bundle.tar    # ~10 min: Python packages + tests
```

## 4. Arm benchmarks for the report (~15–20 min)

```bash
bash scripts/arm_benchmarks.sh
cp -r results/aarch64 /sdcard/Download/qc_results_aarch64
```

Send `Download/qc_results_aarch64` back to the laptop (USB), into `results/aarch64/`.

## 5. The app on the phone

```bash
.venv/bin/python -m qc_assist --source http://127.0.0.1:8080/shot.jpg
```

Open **Chrome on the phone** at http://127.0.0.1:8000 → menu → **Add to Home screen**: QC-Assist
then opens full screen like an app. Point the camera at a board.

- Your own boards: press **Capture 20 good boards** (move the board slightly between captures),
  or `--model` for a VisA model.
- VisA demo scenes (no camera): `.venv/bin/python scripts/demo.py inspect` / `teach`.
- To show the dashboard on the laptop screen instead: add `--host 0.0.0.0` and open
  `http://<phone-ip>:8000` on the laptop (same Wi-Fi).

## Troubleshooting

- `pip` killed / app closes: not enough free RAM. Close other apps; 8 GB phones are safer.
- Camera URL fails: IP Webcam must show "server running"; try http://127.0.0.1:8080 in Chrome.
- Slow first Llama answer: the model loads on first use (~10–20 s on a phone).
- Termux closes by itself / "process completed (signal 9)": Android 12+ kills background
  processes. Set Termux's battery usage to **Unrestricted** (App info → Battery), keep Termux in
  the foreground during setup, and on Android 14+ turn on Developer options → **Disable child
  process restrictions**.
