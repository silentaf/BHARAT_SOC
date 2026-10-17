"""Benchmark the Llama advisory exactly as the dashboard uses it (ExecuTorch TextLLMRunner, in-process).

    python scripts/bench_llama.py                 # laptop or Pi; same fixed prompts on both
    python scripts/bench_llama.py --repeats 3

Runs advice, a shift report and operator questions (English + Hindi) on fixed statistics, so
laptop and Pi numbers are comparable. Writes outputs/benchmarks/llama_<host>.json and a
copy in results/<arch>/ (evidence for the report), including every generated text and
whether it passed the output checks (see qc_assist/advisor.py).
"""

import argparse
import json
import platform
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qc_assist.advisor import LlamaAdvisor  # noqa: E402
from qc_assist.config import load_config  # noqa: E402
from qc_assist.severity import Verdict  # noqa: E402

VERDICT = Verdict(severity="major", score=1.84, area_pct=1.9, bbox=(12, 4, 20, 11), region="top-right")
CRITICAL = Verdict(severity="critical", score=2.6, area_pct=6.4, bbox=(14, 14, 27, 26), region="centre")
STATS = {
    "product": "pcb1", "inspected": 412, "defective": 23, "defect_rate_pct": 5.6,
    "by_severity": {"ok": 389, "minor": 11, "major": 9, "critical": 3},
    "top_defect_regions": [{"region": "top-right", "count": 9}, {"region": "centre", "count": 6},
                           {"region": "bottom-left", "count": 4}],
    "hourly": [{"hour": "06:00", "inspected": 140, "defective": 4},
               {"hour": "07:00", "inspected": 138, "defective": 7},
               {"hour": "08:00", "inspected": 134, "defective": 12}],
    "avg_vision_ms": 33.1,
}
TASKS = [
    ("advise_en", "advise", ("pcb1", VERDICT), {"lang": "en", "recent_defect_rate": 0.056}),
    ("advise_en_critical", "advise", ("pcb1", CRITICAL), {"lang": "en", "recent_defect_rate": 0.12}),
    ("advise_hi", "advise", ("pcb1", VERDICT), {"lang": "hi", "recent_defect_rate": 0.056}),
    ("report_en", "report", (STATS,), {"lang": "en"}),
    ("report_hi", "report", (STATS,), {"lang": "hi"}),
    ("answer_en", "answer", ("Where are most defects this shift, and is it getting worse?", STATS), {"lang": "en"}),
    ("answer_hi", "answer", ("इस शिफ्ट में सबसे ज़्यादा दोष कहाँ हैं?", STATS), {"lang": "hi"}),
    ("answer_en_unknown", "answer", ("Which operator worked the night shift yesterday?", STATS), {"lang": "en"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    adv = LlamaAdvisor.from_config(cfg["llama"], ROOT)
    if not adv.available:
        sys.exit(f"Llama not available: {adv.model} / {adv.tokenizer} (see docs/executorch_llama.md)")

    t0 = time.perf_counter()
    adv.generate("<|begin_of_text|>Hello")  # loads the .pte once, as the dashboard does on first use
    load_s = time.perf_counter() - t0

    runs = []
    for rep in range(args.repeats):
        for name, fn, a, kw in TASKS:
            adv.last_stats = None  # stays None when the answer needs no Llama call (e.g. out-of-scope question)
            t = time.perf_counter()
            res = getattr(adv, fn)(*a, **kw)
            wall = time.perf_counter() - t
            llm = adv.last_stats or {}
            runs.append({"task": name, "repeat": rep, "wall_s": round(wall, 2), "llama_called": bool(llm), **llm,
                         "source": res.source, "rejected_because": res.note, "text": res.text})
            print(f"[{name} #{rep}] {wall:.1f}s  prefill {llm.get('prefill_tok_s')} tok/s  "
                  f"decode {llm.get('decode_tok_s')} tok/s  {res.source} {res.note or ''}\n{res.text}\n",
                  flush=True)

    host = platform.node()
    out = {
        "model": Path(adv.model).name,
        "pte_mb": round(Path(adv.model).stat().st_size / 1e6),
        "machine": platform.machine(), "host": host, "cpus": __import__("os").cpu_count(),
        "max_seq_len": adv.max_seq_len, "max_new_tokens": adv.max_new_tokens, "temperature": adv.temperature,
        "load_plus_first_call_s": round(load_s, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "median": {k: statistics.median(r[k] for r in runs if r["llama_called"])
                   for k in ("prefill_tok_s", "decode_tok_s", "prompt_tokens", "generated_tokens", "wall_s")},
        "advice_wall_s_median": statistics.median(r["wall_s"] for r in runs if r["task"].startswith("advise")),
        "llama_output_accepted": f"{sum(r['source'] == 'llama-executorch' for r in runs)}/"
                                 f"{sum(r['llama_called'] for r in runs)} Llama calls",
        "accepted_by_task": {t: f"{sum(r['source'] == 'llama-executorch' for r in runs if r['task'] == t)}/"
                                f"{sum(r['task'] == t for r in runs)}" for t, *_ in TASKS},
        "runs": runs,
    }
    dst = ROOT / "outputs" / "benchmarks" / f"llama_{host}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    res_dir = ROOT / "results" / platform.machine()
    res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dst, res_dir / dst.name)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    print(f"wrote {dst} and results/{platform.machine()}/{dst.name}")


if __name__ == "__main__":
    main()
