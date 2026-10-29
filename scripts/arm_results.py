"""Put the Arm device measurements (results/aarch64/, from scripts/arm_benchmarks.sh) into the
report and the README, between the <!-- arm-results:begin/end --> markers.

    python scripts/arm_results.py              # rewrite both files
    python scripts/arm_results.py --check      # print the block only

Without results it writes a "pending" note, so it is safe to run at any time.
"""

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "docs" / "technical_report.md", ROOT / "README.md"]
BACKENDS = {"torch": "PyTorch (CPU)", "onnx": "ONNX Runtime (CPU)", "hailo": "Hailo-10H NPU (HailoRT)"}
BEGIN, END = "<!-- arm-results:begin -->", "<!-- arm-results:end -->"


def device_line(res: Path) -> str:
    """Short device description from device_<host>.txt."""
    txt = next(iter(sorted(res.glob("device_*.txt"))), None)
    if txt is None:
        return "Arm device"
    info, cores = {}, []
    for line in txt.read_text().splitlines():
        k, _, v = line.partition(":")
        info.setdefault(k.strip(), v.strip())
        if k.strip() == "Model name":
            cores.append(v.strip())  # one line per core type on big.LITTLE chips
    parts = [info.get("device") or info.get("Model") or info.get("Hardware") or "Arm device"]
    if cores:
        parts.append(" + ".join(sorted(set(cores), reverse=True)))
    if info.get("CPU(s)"):
        parts.append(f"{info['CPU(s)']} cores")
    mem = info.get("MemTotal", "").split()
    if mem and mem[0].isdigit():
        parts.append(f"{int(mem[0]) / 1024 ** 2:.1f} GB RAM visible to Linux")
    if info.get("threads for vision"):
        parts.append(f"vision on {info["threads for vision"]} threads")
    return ", ".join(parts)


def block(res: Path) -> str:
    vision = [json.load(open(p)) for p in sorted(glob.glob(str(res / "pcb*_*.json")))
              if json.load(open(p)).get("backend") in BACKENDS]
    llama = next((json.load(open(p)) for p in sorted(glob.glob(str(res / "llama_*.json")))), None)
    if not vision and not llama:
        return ("**Arm device (Raspberry Pi 5 + AI HAT+ 2):** measurements pending — "
                "`bash scripts/arm_benchmarks.sh` on the Pi ([`docs/pi.md`](docs/pi.md)), "
                "then `python scripts/arm_results.py`.")

    out = [f"**Measured on an Arm device:** {device_line(res)}. Same code and models as the laptop; "
           f"raw JSON in [`results/aarch64/`](results/aarch64/)."]
    if vision:
        out += ["", "| Board | Backend | Image AUROC | Recall | False alarms | ms / frame (mean / p95) | FPS |",
                "|---|---|---|---|---|---|---|"]
        for r in vision:
            out.append(f"| {r['model'][:-3]} | {BACKENDS[r['backend']]} | "
                       f"{r['image_auroc']:.3f} | {r['recall_at_threshold']:.0%} | {r['false_alarm_rate']:.0%} | "
                       f"{r['latency_ms']['mean']:.1f} / {r['latency_ms']['p95']:.1f} | {r['fps']:.1f} |")
        for b in BACKENDS:
            rs = [r for r in vision if r["backend"] == b]
            if rs:
                out.append(f"| **mean** | {BACKENDS[b]} | "
                           f"{statistics.mean(r['image_auroc'] for r in rs):.3f} | | | "
                           f"{statistics.mean(r['latency_ms']['mean'] for r in rs):.1f} | "
                           f"{statistics.mean(r['fps'] for r in rs):.1f} |")
    if llama:
        m = llama["median"]
        out += ["", "| Llama 3.2 1B (ExecuTorch, 8da4w, XNNPACK) on the Arm CPU | Value |", "|---|---|",
                f"| Prefill tokens/s (median) | {m['prefill_tok_s']:.0f} |",
                f"| Decode tokens/s (median) | {m['decode_tok_s']:.1f} |",
                f"| Defect advice, request to text (median) | {llama['advice_wall_s_median']:.1f} s |",
                f"| Load + first call | {llama['load_plus_first_call_s']:.1f} s |",
                f"| Peak RAM (whole process) | {llama['peak_rss_mb']:,} MB |",
                f"| Llama output accepted by the checks | {llama['llama_output_accepted']} |"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "aarch64"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = block(Path(args.results))
    if args.check:
        print(text)
        return
    pat = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.S)
    for f in TARGETS:
        src = f.read_text()
        if not pat.search(src):
            raise SystemExit(f"{f}: markers {BEGIN} ... {END} not found")
        f.write_text(pat.sub(lambda _: f"{BEGIN}\n{text}\n{END}", src))
        print(f"updated {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
