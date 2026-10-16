# Llama 3.2 1B on ExecuTorch (required by PS5)

The challenge requires a Llama-family model for the advisory component, deployed
on-device with **ExecuTorch**. QC-Assist runs the exported `.pte` with ExecuTorch's
`TextLLMRunner` **in-process** (`qc_assist/advisor.py`): the model is loaded once and
stays in memory, so each advisory only pays prefill + decode. The C++ `llama_main`
runner is supported too (`runtime: binary` in `configs/default.yaml`), but it reloads the
model on every call.

Verified with ExecuTorch **1.5.1** (pip wheels exist for x86-64 and for aarch64 /
Raspberry Pi OS, Python 3.11–3.14 — nothing needs to be compiled).

## 1. Get the model (you)

1. On Hugging Face, open `meta-llama/Llama-3.2-1B-Instruct`, accept the Llama 3.2
   Community License and wait for approval. **Do this first — approval can take time.**
2. Create a read token (Settings → Access Tokens), then download only the original
   checkpoint into the project (Hugging Face cache kept inside the project):

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/hf auth login           # paste the token
.venv/bin/hf download meta-llama/Llama-3.2-1B-Instruct \
    --include "original/*" --local-dir models/llama3_2_1b_hf
mkdir -p models/llama3_2_1b
cp models/llama3_2_1b_hf/original/{consolidated.00.pth,params.json,tokenizer.model} models/llama3_2_1b/
```

## 2. Export to `.pte` (laptop)

```bash
.venv/bin/python scripts/export_llama.py
# -> models/llama3_2_1b_8da4w.pte
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

Gotcha (already handled in `advisor.py`): import
`executorch.extension.llm.custom_ops.custom_ops` before creating the runner, otherwise
loading fails with `Missing operator: llama::custom_sdpa.out`.

## 4. On the Raspberry Pi 5

```bash
python3 -m venv .venv && .venv/bin/pip install executorch==1.5.1   # aarch64 wheel
```

Copy `models/llama3_2_1b_8da4w.pte` and `tokenizer.model` across; no export on the Pi.
Arm's Learning Path "Run Llama 3 on a Raspberry Pi 5 using ExecuTorch" covers the
background.

## Pipeline test without Llama weights

Already done on the laptop (2026-10-16): a 2-layer random-weight model with the Llama 3.2
architecture, exported with the same settings, ran through `LlamaAdvisor` (advise,
shift report, Hindi answer) via `TextLLMRunner`, with stats captured. Real-model numbers
go in the report once the weights are available.

## Numbers to record for the report

- `.pte` size; RSS after prefill (the runner logs it)
- prefill and decode tokens/s (`advisor.last_stats`, also shown on the dashboard)
- time from defect detection to advice on screen
- laptop vs Pi 5
