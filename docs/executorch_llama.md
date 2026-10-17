# Llama 3.2 1B on ExecuTorch (required by PS5)

The challenge requires a Llama-family model for the advisory component, deployed
on-device with **ExecuTorch**. QC-Assist runs the exported `.pte` with ExecuTorch's
`TextLLMRunner` **in-process** (`qc_assist/advisor.py`): the model is loaded once and
stays in memory, so each advisory only pays prefill + decode. The C++ `llama_main`
runner is supported too (`runtime: binary` in `configs/default.yaml`), but it reloads the
model on every call.

Verified with ExecuTorch **1.5.1** and PyTorch **2.14.1 (CPU build)** (pip wheels exist for
x86-64 and for aarch64 / Raspberry Pi OS — nothing needs to be compiled).

> ExecuTorch's native kernels are built against one PyTorch version. Installing another
> PyTorch into `.venv` (for example a CUDA build) breaks the export and the runner with
> `undefined symbol ... libcustom_ops_aot_lib.so`. Fix:
> `.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu`

## 1. Get the model (you)

1. On Hugging Face, open `meta-llama/Llama-3.2-1B-Instruct`, accept the Llama 3.2
   Community License and wait for approval. **Do this first — approval can take time.**
2. Create a read token (Settings → Access Tokens), then download only the original
   checkpoint into the project (Hugging Face cache kept inside the project):

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/hf auth login           # paste the token
.venv/bin/hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*"
mkdir -p models/llama3_2_1b
S=$(echo .cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/original)
for f in consolidated.00.pth params.json tokenizer.model; do ln -sf "$PWD/$S/$f" models/llama3_2_1b/$f; done
```

## 2. Export to `.pte` (laptop)

```bash
.venv/bin/python scripts/export_llama.py
# -> models/llama3_2_1b_8da4w.pte  (860 MB; 4 min 46 s and 8.9 GB peak RAM on the laptop)
```

Settings live in `configs/llama_export.yaml`:
- `qmode: 8da4w` — 4-bit grouped weights, 8-bit dynamic activations (XNNPACK; KleidiAI
  kernels on Arm), `embedding_quantize: 4,32`
- KV cache + fused SDPA, 768-token context (advisory prompts are ~150–450 tokens)

`scripts/export_llama.py` does what ExecuTorch's `python -m executorch.extension.llm.export.export_llm`
does, but builds the config directly — Hydra's CLI in that command fails on Python 3.14
(`ValueError: badly formed help string`). It also fixes the output path (ExecuTorch writes
an `output_name` ending in `.pte` relative to the current directory, ignoring `output_dir`).

## 3. Run

`configs/default.yaml` already points at `models/llama3_2_1b_8da4w.pte` and
`models/llama3_2_1b/tokenizer.model`. Once those files exist, the dashboard header shows
**Llama 3.2 · ExecuTorch**, every advisory is logged with `advisory_source = llama-executorch`,
and the dashboard shows decode/prefill tokens/s under each answer.

Gotcha (already handled in `advisor.py`): two kernel libraries must be imported before the
runner loads the model, otherwise `forward` fails to load:
- `executorch.extension.llm.custom_ops.custom_ops`: `Missing operator: llama::custom_sdpa.out`
- `executorch.kernels.quantized`: `kernel 'quantized_decomposed::embedding_4bit.dtype_out' not found`
  (needed because of `embedding_quantize: 4,32`)

## 4. On the Raspberry Pi 5

```bash
python3 -m venv .venv && .venv/bin/pip install executorch==1.5.1   # aarch64 wheel
```

Copy `models/llama3_2_1b_8da4w.pte` and `tokenizer.model` across; no export on the Pi.
Arm's Learning Path "Run Llama 3 on a Raspberry Pi 5 using ExecuTorch" covers the
background.

## How Llama is used (and why)

Measured on the exported model before deciding (prompt experiments, 2026-10-17). Plain
prompts with the statistics as JSON gave confident but wrong output from the 1B model:
it told the operator to stop the line for a *major* defect, called a rising defect rate
"decreasing", pasted raw statistics into the shift report, and looped in Hindi. The
unquantised bf16 model behaved the same way, so this is model size, not quantisation.
What works (all in `qc_assist/advisor.py`):

1. **Code decides.** Severity, the required action and whether to stop the line come from
   a fixed plant policy, written by hand in English and Hindi. Llama never makes that call.
2. **Code computes, Llama phrases.** Counts, rates, the trend and a shift rating (GOOD /
   POOR) are computed from the inspection log and given to Llama as plain English facts,
   with Hindi location names when the reply is in Hindi (Devanagari costs ~3x the tokens,
   and the model reads English facts more reliably).
3. **Worked examples (few-shot).** One or two example exchanges per task. For the shift
   report the example is chosen to match the shift (good or poor); with one fixed example
   the model copied its verdict onto bad shifts.
4. **Every output is checked** before it is shown. It is rejected if it has a number or a
   board location that is not in the facts, contradicts the trend / critical count /
   shift rating, says no defect was found, loops, or is cut off. Rejected output is
   replaced by deterministic text and labelled `template`; the dashboard shows why.
5. **Out of scope.** The log holds boards, not people; questions about operators get a
   fixed answer (the model invented one even with an example of declining).

## Results (laptop, x86-64, `scripts/bench_llama.py --repeats 3`)

From `results/x86_64/llama_VLSI.json` (every generated text is in that file):

| | |
|---|---|
| `.pte` size | 860 MB |
| Load + first call | 6.5 s |
| Peak RAM (whole process) | 1,422 MB |
| Prefill | 244 tokens/s (median) |
| Decode | 31.0 tokens/s (median) |
| Defect advice, end to end | 2.3 s (median) |

| Task | Llama output accepted |
|---|---|
| Advice, English (major / critical) | 3/3, 3/3 |
| Advice, Hindi | 3/3 |
| Shift report, English / Hindi | 3/3, 2/3 (one "3.0%" for a 2.9% rate, caught) |
| Question, English | 3/3 |
| Question, Hindi | not sent to Llama: Hindi figures from the log (its Hindi answers were wrong in earlier runs) |
| Question about people | answered by rule, Llama not called |

Since adding defect types the advice names them ("Look for a missing component in the
top-right corner..."), and the checks also reject a location or defect type paired with
the wrong count, a type not in the facts, and numbers not written exactly as in the facts.

Runner gotcha: `TextLLMRunner` hands each token to Python as UTF-8 text. Llama 3's tokenizer
can split one Devanagari character across two tokens; the binding then raises
`UnicodeDecodeError` and generation stops. `advisor.py` catches it and shows deterministic
text; defect types are given by their English names in Hindi prompts, which avoided it.

Not caught by the checks: wrong ordering words (e.g. "lowest" next to the largest count)
when the numbers themselves are right. Seen in a live dashboard run; noted as a limitation.

## Pipeline test without Llama weights

Done first on the laptop (2026-10-16): a 2-layer random-weight model with the Llama 3.2
architecture, exported with the same settings, ran through `LlamaAdvisor` via
`TextLLMRunner`. Superseded by the real model above.

## Numbers to record on the Pi

```bash
.venv/bin/python scripts/bench_llama.py --repeats 3   # -> results/aarch64/llama_<host>.json
```

- prefill and decode tokens/s, peak RAM, load time, advice latency
- laptop vs Pi 5, and vision ms per frame while Llama runs (CPU contention, see report)
