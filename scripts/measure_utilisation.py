"""Measure hardware utilisation of a command: CPU use, busy cores, memory and threads over time.

    python scripts/measure_utilisation.py vision -- .venv/bin/python scripts/run_inspection.py \
        --model models/pcb1.pt --backend onnx --source data/visa/pcb1/test
    python scripts/measure_utilisation.py llama -- .venv/bin/python scripts/bench_llama.py --repeats 1

Samples the process tree every 0.2 s with psutil. CPU % is of the whole machine (100% = every
logical core busy); "cores busy" is the same number in cores. Writes
results/<arch>/utilisation_<name>_<host>.json.
"""

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
INTERVAL = 0.2


def tree(proc):
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def main():
    name, sep, *cmd = sys.argv[1:]
    assert sep == "--" and cmd, __doc__
    ncpu = psutil.cpu_count()
    psutil.cpu_percent(percpu=True)
    start = time.time()
    child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
    proc = psutil.Process(child.pid)
    seen = {}
    samples = []
    while child.poll() is None:
        time.sleep(INTERVAL)
        cpu = rss = threads = 0.0
        for p in tree(proc):
            try:
                if p.pid not in seen:
                    seen[p.pid] = p
                    p.cpu_percent()
                    continue
                cpu += p.cpu_percent()
                rss += p.memory_info().rss
                threads += p.num_threads()
            except psutil.NoSuchProcess:
                pass
        system = psutil.cpu_percent(percpu=True)
        samples.append({"t": round(time.time() - start, 2), "proc_cores": round(cpu / 100, 2),
                        "rss_mb": round(rss / 2**20), "threads": int(threads),
                        "system_pct": round(sum(system) / len(system), 1)})
    samples = [s for s in samples if s["rss_mb"] > 0]
    cores = [s["proc_cores"] for s in samples]
    active = [c for c in cores if c > 0.5]  # leave out start-up and idle waits
    out = {
        "what": name, "command": " ".join(cmd), "machine": platform.machine(), "host": platform.node(),
        "cpu_model": platform.processor() or "", "logical_cpus": ncpu,
        "total_ram_mb": round(psutil.virtual_memory().total / 2**20),
        "wall_s": round(time.time() - start, 1), "exit_code": child.returncode,
        "cores_busy": {"mean_active": round(statistics.mean(active), 2) if active else 0,
                       "p95": round(sorted(cores)[int(0.95 * (len(cores) - 1))], 2), "max": max(cores)},
        "cpu_pct_of_machine": {"mean_active": round(100 * statistics.mean(active) / ncpu, 1) if active else 0,
                               "max": round(100 * max(cores) / ncpu, 1)},
        "peak_rss_mb": max(s["rss_mb"] for s in samples),
        "peak_rss_pct_of_ram": round(100 * max(s["rss_mb"] for s in samples) * 2**20
                                     / psutil.virtual_memory().total, 1),
        "max_threads": max(s["threads"] for s in samples),
        "other_load_mean_pct": round(statistics.mean(max(s["system_pct"] - 100 * s["proc_cores"] / ncpu, 0)
                                                     for s in samples), 1),
        "samples": samples,
    }
    dest = ROOT / "results" / platform.machine() / f"utilisation_{name}_{platform.node()}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
