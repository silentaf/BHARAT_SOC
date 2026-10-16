"""Operator-facing language features from Llama running on-device through ExecuTorch.

Three jobs, all on the Arm CPU:
  1. advise   — what a detected defect likely is and what the operator should do
  2. report   — end-of-shift quality summary
  3. answer   — operator questions about the inspection log ("how many critical today?")

Numbers always come from the vision pipeline or SQL (inspection_log.summary);
Llama only explains them, so a 1B model never has to invent or compute statistics.

The vision pipeline never waits on the LLM: calls run on a single background worker.

Runtime (see docs/executorch_llama.md), both executing the same exported `.pte`:
  - "python" (default): ExecuTorch's TextLLMRunner in-process. The model is loaded once
    and stays in memory, so each request only pays prefill + decode.
  - "binary": the C++ `llama_main` runner as a subprocess (reloads the model per call).
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .severity import Verdict

LANGUAGES = {"en": "English", "hi": "Hindi (Devanagari script)"}

ADVISE_SYSTEM = (
    "You are a quality-control assistant on a printed-circuit-board inspection line. "
    "Given an automated inspection result, reply in at most 4 short lines: "
    "1) what the defect likely is, 2) probable cause, 3) action for the operator, "
    "4) whether to stop the line. Be concrete. Do not invent measurements."
)
REPORT_SYSTEM = (
    "You are a quality-control assistant. Write a short end-of-shift quality report "
    "for the line supervisor from the statistics given: overall result, notable patterns "
    "(severity, location, time of day), and up to 3 recommended actions. "
    "Use only the numbers provided."
)
ANSWER_SYSTEM = (
    "You are a quality-control assistant. Answer the operator's question using only the "
    "inspection statistics provided. If the statistics do not contain the answer, say so. "
    "Answer in at most 3 sentences."
)


@dataclass
class LLMResult:
    text: str
    source: str  # "llama-executorch" or "fallback"


# Kept for callers that think in terms of advisories.
Advisory = LLMResult


def chat_prompt(system: str, user: str, lang: str = "en") -> str:
    """Llama 3 chat template."""
    if lang != "en":
        system += f" Reply in {LANGUAGES[lang]}."
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_prompt(product: str, verdict: Verdict, recent_defect_rate: float | None = None,
                 defect_type: str | None = None, lang: str = "en") -> str:
    facts = [
        f"Product: {product}",
        f"Severity: {verdict.severity}",
        f"Anomaly score: {verdict.score:.2f} (1.0 = threshold)",
        f"Defect area: {verdict.area_pct:.2f}% of the board, located {verdict.region}",
    ]
    if defect_type:
        facts.append(f"Classifier defect type: {defect_type}")
    if recent_defect_rate is not None:
        facts.append(f"Defect rate over recent inspections: {100 * recent_defect_rate:.1f}%")
    return chat_prompt(ADVISE_SYSTEM, "Inspection result:\n" + "\n".join(facts), lang)


def fallback_advice(product: str, verdict: Verdict) -> str:
    """Used only when the Llama runtime is not configured yet. Clearly logged as such."""
    action = {
        "minor": "Mark the board for manual re-check; keep the line running.",
        "major": "Remove the board and inspect the upstream station.",
        "critical": "Stop the line and call the shift supervisor.",
    }[verdict.severity]
    return f"{verdict.severity.upper()} anomaly on {product} ({verdict.region}, {verdict.area_pct:.1f}% area). {action}"


def fallback_report(stats: dict) -> str:
    s = stats["by_severity"]
    lines = [
        f"{stats['product']}: {stats['inspected']} inspected, {stats['defective']} defective "
        f"({stats['defect_rate_pct']}%).",
        f"Severity — critical {s['critical']}, major {s['major']}, minor {s['minor']}.",
    ]
    if stats["top_defect_regions"]:
        top = ", ".join(f"{r['region']} ({r['count']})" for r in stats["top_defect_regions"])
        lines.append(f"Most frequent defect locations: {top}.")
    return " ".join(lines)


class LlamaAdvisor:
    def __init__(self, runner: str | None, model: str | None, tokenizer: str | None,
                 max_seq_len: int = 512, temperature: float = 0.2, timeout_s: float = 120,
                 runtime: str = "python", max_new_tokens: int = 160):
        self.runner, self.model, self.tokenizer = runner, model, tokenizer
        self.max_seq_len, self.temperature, self.timeout_s = max_seq_len, temperature, timeout_s
        self.runtime, self.max_new_tokens = runtime, max_new_tokens
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._et_runner = None
        self.last_stats: dict | None = None  # prefill/decode speed of the latest call, for the report

    @classmethod
    def from_config(cls, llama_cfg: dict, root: Path | None = None) -> "LlamaAdvisor":
        def resolve(p):
            if not p:
                return None
            p = Path(p)
            return str(p if p.is_absolute() or root is None else root / p)
        return cls(resolve(llama_cfg.get("runner")), resolve(llama_cfg.get("model")),
                   resolve(llama_cfg.get("tokenizer")), llama_cfg.get("max_seq_len", 512),
                   llama_cfg.get("temperature", 0.2), runtime=llama_cfg.get("runtime", "python"),
                   max_new_tokens=llama_cfg.get("max_new_tokens", 160))

    @property
    def available(self) -> bool:
        if not all(p and Path(p).exists() for p in (self.model, self.tokenizer)):
            return False
        if self.runtime == "binary":
            return bool(self.runner and Path(self.runner).exists())
        try:
            from executorch.extension.llm.runner import TextLLMRunner  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, prompt: str) -> str:
        return self._generate_python(prompt) if self.runtime == "python" else self._generate_binary(prompt)

    def _generate_python(self, prompt: str) -> str:
        # Registers ExecuTorch's LLM kernels (llama::custom_sdpa, update_cache) used by the exported
        # KV-cache attention; without it loading 'forward' fails with "Missing operator".
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
        if self._et_runner is None:
            self._et_runner = TextLLMRunner(self.model, self.tokenizer)  # loaded once, kept in memory
        pieces, stats = [], {}

        def on_stats(s):
            stats.update(prompt_tokens=s.num_prompt_tokens, generated_tokens=s.num_generated_tokens,
                         prefill_ms=s.prompt_eval_end_ms - s.inference_start_ms,
                         decode_ms=s.inference_end_ms - s.first_token_ms,
                         total_ms=s.inference_end_ms - s.inference_start_ms)

        self._et_runner.reset()  # every prompt is self-contained; start from an empty KV cache
        cfg = GenerationConfig(echo=False, max_new_tokens=self.max_new_tokens, seq_len=self.max_seq_len,
                               temperature=self.temperature)
        self._et_runner.generate(prompt, cfg, token_callback=pieces.append, stats_callback=on_stats)
        if stats:
            stats["prefill_tok_s"] = round(1000 * stats["prompt_tokens"] / max(stats["prefill_ms"], 1), 1)
            stats["decode_tok_s"] = round(1000 * max(stats["generated_tokens"] - 1, 0) / max(stats["decode_ms"], 1), 1)
            self.last_stats = stats
        return _strip_runner_output("".join(pieces), prompt)

    def _generate_binary(self, prompt: str) -> str:
        cmd = [
            self.runner,
            f"--model_path={self.model}",
            f"--tokenizer_path={self.tokenizer}",
            f"--prompt={prompt}",
            f"--seq_len={self.max_seq_len}",
            f"--temperature={self.temperature}",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, check=True).stdout
        return _strip_runner_output(out, prompt)

    def advise(self, product: str, verdict: Verdict, lang: str = "en", **ctx) -> LLMResult:
        if not self.available:
            return LLMResult(fallback_advice(product, verdict), "fallback")
        return LLMResult(self.generate(build_prompt(product, verdict, lang=lang, **ctx)), "llama-executorch")

    def report(self, stats: dict, lang: str = "en") -> LLMResult:
        if not self.available:
            return LLMResult(fallback_report(stats), "fallback")
        user = "Shift statistics (JSON):\n" + json.dumps(stats, ensure_ascii=False)
        return LLMResult(self.generate(chat_prompt(REPORT_SYSTEM, user, lang)), "llama-executorch")

    def answer(self, question: str, stats: dict, lang: str = "en") -> LLMResult:
        if not self.available:
            return LLMResult("Llama is not configured yet. Current statistics: " + fallback_report(stats), "fallback")
        user = f"Inspection statistics (JSON):\n{json.dumps(stats, ensure_ascii=False)}\n\nQuestion: {question}"
        return LLMResult(self.generate(chat_prompt(ANSWER_SYSTEM, user, lang)), "llama-executorch")

    def advise_async(self, product: str, verdict: Verdict, **ctx) -> Future:
        return self._pool.submit(self.advise, product, verdict, **ctx)

    def submit(self, fn_name: str, *args, **kwargs) -> Future:
        return self._pool.submit(getattr(self, fn_name), *args, **kwargs)


def _strip_runner_output(out: str, prompt: str) -> str:
    # The runner echoes the prompt before the generated text and may append stats.
    if prompt in out:
        out = out.split(prompt, 1)[1]
    for marker in ("<|eot_id|>", "PyTorchObserver"):
        out = out.split(marker, 1)[0]
    return out.strip()
