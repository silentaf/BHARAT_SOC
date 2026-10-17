"""Operator-facing language features from Llama running on-device through ExecuTorch.

Three jobs, all on the Arm CPU:
  1. advise   — what to look for at a detected defect and its usual cause
  2. report   — end-of-shift quality note for the supervisor
  3. answer   — operator questions about the inspection log ("how many critical today?")

Division of labour (measured on Llama 3.2 1B, see docs/executorch_llama.md):
  - Code decides. Severity, the required action and whether to stop the line come from a
    fixed plant policy, written in English and Hindi. A 1B model must not make that call:
    in our tests it told the operator to stop the line for a major defect.
  - Code computes. Counts, rates, the trend and the shift rating come from the vision
    pipeline / SQL (inspection_log.summary) and reach Llama as plain-language facts, not JSON.
  - Llama explains, guided by worked examples (few-shot), in the operator's language.
  - Code checks. Output that contains a number not in the facts, contradicts them (e.g.
    "stable" when the defect rate is rising) or loops is replaced by the deterministic text
    and labelled "template".

The vision pipeline never waits on the LLM: calls run on a single background worker.

Runtime (see docs/executorch_llama.md), both executing the same exported `.pte`:
  - "python" (default): ExecuTorch's TextLLMRunner in-process. The model is loaded once
    and stays in memory, so each request only pays prefill + decode.
  - "binary": the C++ `llama_main` runner as a subprocess (reloads the model per call).
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .severity import Verdict

LANGUAGES = {"en": "English", "hi": "हिंदी"}

# ---- plant policy (deterministic, never generated) ----------------------------------------

SEVERITY_NAME = {
    "en": {"minor": "MINOR", "major": "MAJOR", "critical": "CRITICAL"},
    "hi": {"minor": "मामूली", "major": "बड़ा", "critical": "गंभीर"},
}
ACTION = {
    "en": {
        "minor": "Mark the board for manual re-check. Keep the line running.",
        "major": "Remove the board and check the upstream station. Keep the line running.",
        "critical": "Stop the line and call the shift supervisor.",
    },
    "hi": {
        "minor": "बोर्ड को दोबारा हाथ से जाँच के लिए चिह्नित करें। लाइन चालू रखें।",
        "major": "बोर्ड हटाएँ और पिछले स्टेशन की जाँच करें। लाइन चालू रखें।",
        "critical": "लाइन रोकें और शिफ्ट सुपरवाइज़र को बुलाएँ।",
    },
}
REGION_HI = {
    "top-left": "ऊपर-बाएँ", "top-centre": "ऊपर-बीच", "top-right": "ऊपर-दाएँ",
    "middle-left": "बीच-बाएँ", "centre": "बीच में", "middle-right": "बीच-दाएँ",
    "bottom-left": "नीचे-बाएँ", "bottom-centre": "नीचे-बीच", "bottom-right": "नीचे-दाएँ",
}
DEFECT_WORD = {"en": "defect", "hi": "दोष"}


def region_text(region: str, lang: str) -> str:
    return REGION_HI.get(region, region) if lang == "hi" else region


def policy_line(verdict: Verdict, lang: str = "en") -> str:
    """First line of every advisory: severity, location and the required action."""
    sev = SEVERITY_NAME[lang][verdict.severity]
    where = region_text(verdict.region, lang)
    return f"{sev} {DEFECT_WORD[lang]} · {where} · {verdict.area_pct:.1f}% — {ACTION[lang][verdict.severity]}"


# ---- facts given to Llama ----------------------------------------------------------------

def trend(stats: dict) -> tuple[str, float | None, float | None]:
    """'rising' / 'falling' / 'stable' / 'unknown', from first- vs last-hour defect rate."""
    hours = [h for h in stats.get("hourly", []) if h["inspected"]]
    if len(hours) < 2:
        return "unknown", None, None
    first = 100 * hours[0]["defective"] / hours[0]["inspected"]
    last = 100 * hours[-1]["defective"] / hours[-1]["inspected"]
    if abs(last - first) < max(1.0, 0.25 * first):
        return "stable", first, last
    return ("rising" if last > first else "falling"), first, last


TREND_WORD = {
    "en": {"rising": "RISING", "falling": "FALLING", "stable": "STABLE", "unknown": "not enough data"},
    "hi": {"rising": "बढ़ रहा", "falling": "घट रहा", "stable": "स्थिर", "unknown": "पर्याप्त डेटा नहीं"},
}


def verdict_facts(product: str, verdict: Verdict, lang: str = "en", recent_defect_rate: float | None = None,
                  defect_type: str | None = None) -> str:
    sev = SEVERITY_NAME[lang][verdict.severity].lower()
    where = region_text(verdict.region, lang)
    if lang == "hi":
        facts = [f"बोर्ड: {product}.", f"परिणाम: {sev} दोष.", f"जगह: {where}.", f"आकार: बोर्ड का {verdict.area_pct:.1f}%."]
        if defect_type:
            facts.append(f"दोष का प्रकार: {defect_type}.")
    else:
        facts = [f"Board: {product}.", f"Result: {sev} defect.", f"Location: {where}.",
                 f"Size: {verdict.area_pct:.1f}% of the board."]
        if defect_type:
            facts.append(f"Defect type: {defect_type}.")
        if recent_defect_rate is not None:
            facts.append(f"Defect rate over recent boards: {100 * recent_defect_rate:.1f}%.")
    return " ".join(facts)


def shift_rating(stats: dict) -> str:
    """'poor' if there were critical defects or the defect rate is rising, else 'good'."""
    return "poor" if stats["by_severity"]["critical"] or trend(stats)[0] == "rising" else "good"


def stats_facts(stats: dict, lang: str = "en") -> str:
    """Shift statistics as plain English facts, whatever the reply language: the 1B model reads
    English facts more reliably, and Devanagari costs ~3x the tokens. For Hindi replies the
    location names are given in Hindi too, so the model copies them instead of translating."""
    s = stats["by_severity"]
    t, first, last = trend(stats)

    def where(region):
        return f"{region} ({REGION_HI[region]})" if lang == "hi" and region in REGION_HI else region

    parts = [f"Product: {stats['product']}.", f"Inspected: {stats['inspected']}.",
             f"Defective: {stats['defective']} ({stats['defect_rate_pct']}%).",
             f"Severity: {s['critical']} critical, {s['major']} major, {s['minor']} minor."]
    if stats.get("top_defect_regions"):
        parts.append("Defects by location: " + ", ".join(
            f"{where(r['region'])} {r['count']}" for r in stats["top_defect_regions"]) + ".")
    hourly = [h for h in stats.get("hourly", []) if h["inspected"]]
    if hourly:
        parts.append("Defect rate by hour: " + ", ".join(
            f"{h['hour']} {100 * h['defective'] / h['inspected']:.1f}%" for h in hourly) + ".")
    parts.append(f"Trend: {TREND_WORD['en'][t]}"
                 + (f" ({first:.1f}% first hour, {last:.1f}% last hour)." if first is not None else "."))
    if shift_rating(stats) == "poor":
        why = [f"{s['critical']} critical defects"] if s["critical"] else []
        why += ["rising defect rate"] if t == "rising" else []
        parts.append(f"Shift rating: POOR ({', '.join(why)}).")
    else:
        parts.append("Shift rating: GOOD (no critical defects, defect rate not rising).")
    if stats.get("recent_defects"):
        parts.append("Recent defects: " + ", ".join(
            f"#{d['id']} {d['severity']} {where(d['region'])}" for d in stats["recent_defects"][:5]) + ".")
    return " ".join(parts)


# Statistics behind the worked examples; their facts are rendered with stats_facts() so the
# examples always match the format of the real prompt.
EXAMPLE_GOOD = {
    "product": "pcb4", "inspected": 380, "defective": 8, "defect_rate_pct": 2.1,
    "by_severity": {"ok": 372, "minor": 6, "major": 2, "critical": 0},
    "top_defect_regions": [{"region": "centre", "count": 5}, {"region": "top-left", "count": 2}],
    "hourly": [{"hour": "14:00", "inspected": 200, "defective": 4}, {"hour": "15:00", "inspected": 180, "defective": 4}],
}
EXAMPLE_POOR = {
    "product": "pcb2", "inspected": 290, "defective": 31, "defect_rate_pct": 10.7,
    "by_severity": {"ok": 259, "minor": 15, "major": 12, "critical": 4},
    "top_defect_regions": [{"region": "bottom-right", "count": 14}, {"region": "top-left", "count": 6}],
    "hourly": [{"hour": "22:00", "inspected": 150, "defective": 9}, {"hour": "23:00", "inspected": 140, "defective": 22}],
}


# ---- prompts: system message + worked examples, per task and language ----------------------

PROMPTS = {
    "advise": {
        "en": ("You help operators on a printed-circuit-board assembly line. Reply with one or two short "
               "sentences: what to look for at the defect location and the usual cause. Never decide the "
               "action or about stopping the line.",
               [("Board: pcb2. Result: minor defect. Location: bottom-left. Size: 0.6% of the board.",
                 "Look for a small scratch or flux residue near the bottom-left components. This is usually "
                 "handling damage or incomplete cleaning after soldering."),
                ("Board: pcb4. Result: critical defect. Location: centre. Size: 4.2% of the board.",
                 "Look for a missing or shifted component in the centre of the board. This is often caused by "
                 "a pick-and-place nozzle or feeder problem.")]),
        "hi": ("आप एक PCB असेंबली लाइन पर ऑपरेटर की मदद करते हैं। एक या दो छोटे वाक्यों में सरल हिंदी में "
               "बताइए कि दोष वाली जगह पर क्या देखना है और इसका सामान्य कारण क्या है।",
               [("बोर्ड: pcb2. परिणाम: मामूली दोष. जगह: नीचे-बाएँ. आकार: बोर्ड का 0.6%.",
                 "नीचे-बाएँ हिस्से के पुर्ज़ों के पास छोटी खरोंच या फ्लक्स का अवशेष देखें। यह आमतौर पर "
                 "हैंडलिंग से नुकसान या सोल्डरिंग के बाद अधूरी सफ़ाई से होता है।"),
                ("बोर्ड: pcb4. परिणाम: गंभीर दोष. जगह: बीच में. आकार: बोर्ड का 4.2%.",
                 "बीच के हिस्से में कोई गायब या खिसका हुआ पुर्ज़ा देखें। यह अक्सर पिक-एंड-प्लेस मशीन की "
                 "नोज़ल की गड़बड़ी से होता है।")]),
    },
}

REPORT_SYSTEM = ("You write a short end-of-shift note for the line supervisor from the given facts. Write 3 "
                 "sentences: the shift rating and trend, where defects cluster, one recommended action. Copy "
                 "numbers and the rating exactly; do not calculate.")
ANSWER_SYSTEM = ("Answer the operator's question in at most 2 sentences using only the facts given. Copy numbers "
                 "exactly. If the facts do not answer it, say so.")
IN_HINDI = " Reply in simple Hindi (Devanagari script)."

# Worked examples. The report example is picked to match the shift (good or poor): with a single
# fixed example the 1B model copied its verdict ("good shift, stable") onto bad shifts.
REPORT_EXAMPLES = {
    ("good", "en"): "pcb4 had a good shift: 8 of 380 boards were defective (2.1%), with no critical defects and a "
                    "stable defect rate. Most defects were in the centre (5). Check the components in the centre "
                    "of the board at the start of the next shift.",
    ("poor", "en"): "pcb2 had a poor shift: 31 of 290 boards were defective (10.7%), including 4 critical defects, "
                    "and the defect rate rose from 6.0% to 15.7%. Most defects were in the bottom-right (14). "
                    "Inspect the station that places the bottom-right components before the next shift.",
    ("good", "hi"): "pcb4 की शिफ्ट अच्छी रही: 380 में से 8 बोर्ड दोषपूर्ण थे (2.1%), कोई गंभीर दोष नहीं था और दोष "
                    "दर स्थिर रही। ज़्यादातर दोष बीच में थे (5)। अगली शिफ्ट की शुरुआत में बोर्ड के बीच वाले "
                    "पुर्ज़ों की जाँच करें।",
    ("poor", "hi"): "pcb2 की शिफ्ट खराब रही: 290 में से 31 बोर्ड दोषपूर्ण थे (10.7%), जिनमें 4 गंभीर दोष थे, और "
                    "दोष दर 6.0% से बढ़कर 15.7% हो गई। ज़्यादातर दोष नीचे-दाएँ थे (14)। अगली शिफ्ट से पहले "
                    "नीचे-दाएँ पुर्ज़े लगाने वाले स्टेशन की जाँच करें।",
}
ANSWER_EXAMPLES = {
    "en": [("Where are most defects?", "Most defects are in the bottom-right (14), followed by the top-left (6)."),
           ("Is the defect rate going up?", "Yes, the defect rate is rising: 6.0% in the first hour and 15.7% in "
                                            "the last hour.")],
    "hi": [("सबसे ज़्यादा दोष कहाँ हैं?", "सबसे ज़्यादा दोष नीचे-दाएँ हैं (14), उसके बाद ऊपर-बाएँ (6)।"),
           ("क्या दोष दर बढ़ रही है?", "हाँ, दोष दर बढ़ रही है: पहले घंटे 6.0% और आख़िरी घंटे 15.7%।")],
}
QA_LABELS = {"en": ("Facts", "Question"), "hi": ("Facts", "सवाल")}


def report_prompt(stats: dict, lang: str = "en") -> tuple[str, str]:
    """(prompt, facts) for the shift report."""
    lang = lang if lang in LANGUAGES else "en"
    rating = shift_rating(stats)
    example = EXAMPLE_POOR if rating == "poor" else EXAMPLE_GOOD
    facts = stats_facts(stats, lang)
    system = REPORT_SYSTEM + (IN_HINDI if lang == "hi" else "")
    shots = [(stats_facts(example, lang), REPORT_EXAMPLES[(rating, lang)])]
    return chat_prompt(system, facts, shots), facts


def answer_prompt(question: str, stats: dict, lang: str = "en") -> tuple[str, str]:
    """(prompt, facts) for an operator question."""
    lang = lang if lang in LANGUAGES else "en"
    f_label, q_label = QA_LABELS[lang]
    facts = stats_facts(stats, lang)
    system = ANSWER_SYSTEM + (IN_HINDI if lang == "hi" else "")
    # One example conversation: facts once, then several questions (keeps the prompt short).
    (q0, a0), *more = ANSWER_EXAMPLES[lang]
    shots = [(f"{f_label}: {stats_facts(EXAMPLE_POOR, lang)}\n\n{q_label}: {q0}", a0)]
    shots += [(f"{q_label}: {q}", a) for q, a in more]
    return chat_prompt(system, f"{f_label}: {facts}\n\n{q_label}: {question}", shots), facts


def chat_prompt(system: str, user: str, shots=()) -> str:
    """Llama 3 chat template, with optional worked examples as earlier turns."""
    def turn(role, text):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
    p = "<|begin_of_text|>" + turn("system", system)
    for u, a in shots:
        p += turn("user", u) + turn("assistant", a)
    return p + turn("user", user) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def task_prompt(task: str, user: str, lang: str = "en") -> str:
    system, shots = PROMPTS[task][lang if lang in PROMPTS[task] else "en"]
    return chat_prompt(system, user, shots)


def build_prompt(product: str, verdict: Verdict, recent_defect_rate: float | None = None,
                 defect_type: str | None = None, lang: str = "en") -> str:
    return task_prompt("advise", verdict_facts(product, verdict, lang, recent_defect_rate, defect_type), lang)


# ---- deterministic text (Llama not configured, or its output rejected) ---------------------

def fallback_advice(product: str, verdict: Verdict, lang: str = "en") -> str:
    return policy_line(verdict, lang)


def fallback_report(stats: dict, lang: str = "en") -> str:
    s = stats["by_severity"]
    t, first, last = trend(stats)
    top = ", ".join(f"{region_text(r['region'], lang)} ({r['count']})" for r in stats.get("top_defect_regions", []))
    tr = TREND_WORD[lang][t] + (f" ({first:.1f}% → {last:.1f}%)" if first is not None else "")
    if lang == "hi":
        lines = [f"{stats['product']}: {stats['inspected']} जाँचे गए, {stats['defective']} दोषपूर्ण "
                 f"({stats['defect_rate_pct']}%).",
                 f"गंभीरता — गंभीर {s['critical']}, बड़े {s['major']}, मामूली {s['minor']}।",
                 f"रुझान: {tr}।"]
        if top:
            lines.append(f"सबसे ज़्यादा दोष: {top}।")
    else:
        lines = [f"{stats['product']}: {stats['inspected']} inspected, {stats['defective']} defective "
                 f"({stats['defect_rate_pct']}%).",
                 f"Severity — critical {s['critical']}, major {s['major']}, minor {s['minor']}.",
                 f"Trend: {tr.lower()}."]
        if top:
            lines.append(f"Most frequent defect locations: {top}.")
    return " ".join(lines)


# The log holds boards, not people. Asked "which operator worked last night?", the 1B model
# invented an answer even with an example of declining, so such questions never reach it.
_ABOUT_PEOPLE = re.compile(r"\b(who|whom|operators?|persons?|people|workers?|staff|names?)\b|कौन|ऑपरेटर|कर्मचारी|व्यक्ति|नाम",
                           re.I)
OUT_OF_SCOPE = {
    "en": "The inspection log records boards, defects, severity, location and time, not people, so I cannot answer that.",
    "hi": "जाँच लॉग में बोर्ड, दोष, गंभीरता, जगह और समय दर्ज होते हैं, लोग नहीं, इसलिए मैं इसका जवाब नहीं दे सकता।",
}

NO_ANSWER = {"en": "Llama could not answer this reliably from the log. Current figures: ",
             "hi": "Llama लॉग से इसका भरोसेमंद जवाब नहीं दे सका। अभी के आँकड़े: "}


# ---- output checks -------------------------------------------------------------------------

_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUM.findall(text.translate(_DEVANAGARI_DIGITS))}


# Words that contradict a fact (English matched lower-case). A 1B model copies wording from its
# examples, e.g. "good shift, no critical defects, stable" for a shift that was none of those.
CONTRADICTS = {
    "rising": ("stable", "steady", "decreas", "falling", "improv", "not rising", "not increasing",
               "not getting worse", "isn't rising", "स्थिर", "घट", "नहीं बढ़", "बढ़ नहीं"),
    "falling": ("rising", "increas", "worse", "बढ़"),
    "stable": ("rising", "increas", "decreas", "falling", "बढ़", "घट"),
    "unknown": ("rising", "increas", "decreas", "falling", "worse", "improv", "बढ़", "घट"),  # < 2 hours of data
    "critical": ("no critical", "कोई गंभीर दोष नहीं"),
    "bad_shift": ("good shift", "success", "satisfactory", "अच्छी"),
    "defect_present": ("no defect", "not found", "nothing", "नहीं देखा", "नहीं मिला", "कोई दोष नहीं"),
}


# Every board location in English and Hindi, longest first so "top-right" is not also read as "right".
_LOCATIONS = sorted([(r, r) for r in REGION_HI] + [(r, h) for r, h in REGION_HI.items()] + [("centre", "center")],
                    key=lambda x: -len(x[1]))


def _locations(text: str) -> set[str]:
    found, low = set(), text.lower()
    for region, name in _LOCATIONS:
        if name in low:
            found.add(region)
            low = low.replace(name, " ")
    return found


def check_output(text: str, facts: str, stats: dict | None = None, defect_present: bool = False) -> str | None:
    """Reason to reject a generation, or None if it is usable."""
    if not text.strip():
        return "empty"
    if text.rstrip()[-1] not in ".!?।)":
        return "incomplete (token limit reached)"
    extra = _numbers(text) - _numbers(facts)
    if extra:
        return "numbers not in the facts: " + ", ".join(f"{n:g}" for n in sorted(extra))
    extra_loc = _locations(text) - _locations(facts)
    if extra_loc:
        return "locations not in the facts: " + ", ".join(sorted(extra_loc))
    if defect_present:
        hit = next((w for w in CONTRADICTS["defect_present"] if w in text.lower()), None)
        if hit:
            return f"contradicts the facts (defect_present): '{hit}'"
    if stats:
        t = trend(stats)[0]
        rules = [t] if t in CONTRADICTS else []
        if stats["by_severity"]["critical"]:
            rules.append("critical")
        if stats["by_severity"]["critical"] or t == "rising":
            rules.append("bad_shift")
        low = text.lower()
        for rule in rules:
            hit = next((w for w in CONTRADICTS[rule] if w in low), None)
            if hit:
                return f"contradicts the facts ({rule}): '{hit}'"
    words = text.split()
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    if grams and grams.most_common(1)[0][1] >= 3:
        return "repetition"
    return None


def clean_output(text: str) -> str:
    """Drop a 'Here is ...:' preamble and an unfinished last sentence (max_new_tokens reached)."""
    lines = text.strip().splitlines()
    if lines and re.match(r"^(here (is|are)|sure|certainly)\b.*:\s*$", lines[0].strip(), re.I):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    if text and text[-1] not in ".!?।)":
        cut = max(text.rfind(c) for c in ".!?।")
        if cut > 0:
            text = text[:cut + 1]
    return text


@dataclass
class LLMResult:
    text: str
    source: str              # "llama-executorch", "template" (Llama output rejected) or "fallback" (no Llama)
    note: str | None = None  # why a generation was rejected


# Kept for callers that think in terms of advisories.
Advisory = LLMResult


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

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        if self.runtime == "python":
            return self._generate_python(prompt, max_new_tokens or self.max_new_tokens)
        return self._generate_binary(prompt)

    def _generate_python(self, prompt: str, max_new_tokens: int) -> str:
        # Register the kernels the exported graph uses outside XNNPACK; without them loading 'forward'
        # fails with "Missing operator" / "kernel ... not found":
        #   custom_ops: llama::custom_sdpa, update_cache (KV-cache attention)
        #   quantized:  quantized_decomposed::embedding_4bit (embedding_quantize 4,32)
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        from executorch.kernels import quantized  # noqa: F401
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
        cfg = GenerationConfig(echo=False, max_new_tokens=max_new_tokens, seq_len=self.max_seq_len,
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

    def _checked(self, task: str, facts: str, lang: str, max_new_tokens: int) -> tuple[str | None, str | None]:
        """(text, None) if Llama's output passes the checks, else (None, reason)."""
        text = clean_output(self.generate(task_prompt(task, facts, lang), max_new_tokens))
        reason = check_output(text, facts, defect_present=task == "advise")
        return (None, reason) if reason else (text, None)

    def advise(self, product: str, verdict: Verdict, lang: str = "en", recent_defect_rate: float | None = None,
               defect_type: str | None = None) -> LLMResult:
        head = policy_line(verdict, lang)
        if not self.available:
            return LLMResult(head, "fallback")
        facts = verdict_facts(product, verdict, lang, recent_defect_rate, defect_type)
        text, why = self._checked("advise", facts, lang, 90)
        return LLMResult(f"{head}\n{text}", "llama-executorch") if text else LLMResult(head, "template", why)

    def report(self, stats: dict, lang: str = "en") -> LLMResult:
        if not self.available or not stats["inspected"]:
            return LLMResult(fallback_report(stats, lang), "fallback")
        prompt, facts = report_prompt(stats, lang)
        text = clean_output(self.generate(prompt, 140))
        why = check_output(text, facts, stats)
        return LLMResult(fallback_report(stats, lang), "template", why) if why else LLMResult(text, "llama-executorch")

    def answer(self, question: str, stats: dict, lang: str = "en") -> LLMResult:
        if _ABOUT_PEOPLE.search(question):
            return LLMResult(OUT_OF_SCOPE.get(lang, OUT_OF_SCOPE["en"]), "template", "question about people")
        if not self.available:
            return LLMResult(NO_ANSWER[lang] + fallback_report(stats, lang), "fallback")
        prompt, facts = answer_prompt(question, stats, lang)
        text = clean_output(self.generate(prompt, 110))
        why = check_output(text, facts + " " + question, stats)
        if why:
            return LLMResult(NO_ANSWER[lang] + fallback_report(stats, lang), "template", why)
        return LLMResult(text, "llama-executorch")

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
