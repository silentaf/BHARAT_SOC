# Running QC-Assist on a Raspberry Pi 5 + AI HAT+ 2

Vision CNN on the **Hailo-10H NPU** (HailoRT), nearest-neighbour search and **Llama 3.2 1B
(ExecuTorch)** on the Pi's Arm Cortex-A76 CPU. The camera is a phone running the IP Webcam app
on the same Wi-Fi (a USB webcam also works: `--source 0`).

**You need:** Pi 5 (8 GB), AI HAT+ 2, active cooler, official 27 W USB-C supply, microSD card
(32 GB+), a phone with IP Webcam, and the laptop on the same Wi-Fi. Time: about 2 hours.

## 1. Laptop: flash the card, make the bundle

1. Install **Raspberry Pi Imager** (https://www.raspberrypi.com/software/). Choose
   *Raspberry Pi 5* → **Raspberry Pi OS (64-bit)** → the microSD card.
2. In *Edit settings*: hostname `qcpi`, a user name and password, your Wi-Fi, and on the
   *Services* tab **Enable SSH** (password). Write the card.
3. In the project folder: `bash scripts/make_bundle.sh` → `outputs/qc_assist_bundle.tar`
   (models, Hailo HEFs, test data; ~1.2 GB).

## 2. Assemble and boot

Fit the active cooler, then the AI HAT+ 2 on the GPIO header with its ribbon cable in the Pi's
PCIe connector (follow the leaflet in the box). Insert the card, power on, wait ~2 minutes.

From the laptop:

```bash
ssh <user>@qcpi.local
```

## 3. On the Pi: firmware and the Hailo driver

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot
```

SSH in again, then:

```bash
sudo apt install -y dkms git
sudo apt install -y hailo-h10-all      # AI HAT+ 2 only (hailo-all is for the older HAT)
sudo reboot
```

SSH in again and check the NPU:

```bash
hailortcli fw-control identify         # must print the firmware version; "N/A" names are normal
hailortcli --version                   # note this version for the report
```

## 4. Copy the bundle and install QC-Assist

From the **laptop** (a second terminal):

```bash
scp outputs/qc_assist_bundle.tar <user>@qcpi.local:~
```

On the **Pi**:

```bash
git clone https://github.com/silentaf/BHARAT_SOC.git && cd BHARAT_SOC
bash scripts/setup_arm.sh ~/qc_assist_bundle.tar     # ~10 min; must print "HailoRT found"
.venv/bin/python -c "from qc_assist.anomaly import PatchCore; PatchCore.load('models/pcb1.pt', backend='hailo'); print('NPU OK')"
```

If the last line fails with a HEF version error, the Pi's HailoRT is older than the Dataflow
Compiler that built the HEFs (5.4). Send the error and the `hailortcli --version` output back:
the HEFs get recompiled with the matching compiler on the laptop.

## 5. Benchmarks for the report (~30 min)

Leave the Pi idle (no browser, no other jobs):

```bash
DEVICE_NAME="Raspberry Pi 5 (8 GB) + AI HAT+ 2" bash scripts/arm_benchmarks.sh
vcgencmd pmic_read_adc > results/aarch64/power_idle.txt     # Pi 5 rail currents/voltages
```

This measures all four boards with PyTorch, ONNX Runtime **and the Hailo NPU**, plus Llama.
Copy the results to the laptop (run on the **laptop**, in the project folder):

```bash
scp -r <user>@qcpi.local:~/BHARAT_SOC/results/aarch64 results/
python scripts/arm_results.py        # Pi numbers into the report and README
python scripts/build_report.py       # report PDF
```

## 6. The app (live demo)

On the phone: IP Webcam → **Start server**; it shows an address like `http://192.168.1.23:8080`.
On the Pi:

```bash
.venv/bin/python -m qc_assist --source http://192.168.1.23:8080/shot.jpg \
    --model models/pcb4.pt --backend hailo --host 0.0.0.0
```

Open **http://qcpi.local:8000** in the laptop's browser. For your own boards press
**Capture 20 good boards**. VisA scenes without a camera: `.venv/bin/python scripts/demo.py inspect`.

## Troubleshooting

- `ssh: Could not resolve hostname qcpi.local`: find the Pi's IP in your router's device list
  and use it instead.
- `hailortcli` finds no device: reseat the PCIe ribbon cable, check `dmesg | grep -i hailo`.
- Slow first Llama answer: the model loads on first use (~10–15 s on the Pi).
- Hot Pi throttles (`vcgencmd get_throttled` not `0x0`): the active cooler is required.
