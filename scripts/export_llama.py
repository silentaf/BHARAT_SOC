"""Export Llama 3.2 1B Instruct to an ExecuTorch .pte (XNNPACK, 8da4w) for the Arm CPU.

    python scripts/export_llama.py                       # uses configs/llama_export.yaml
    python scripts/export_llama.py export.max_seq_length=1024 export.max_context_length=1024

Same as ExecuTorch's `python -m executorch.extension.llm.export.export_llm`, but builds
the config directly instead of through Hydra's CLI, which breaks on Python 3.14.
Run it on the laptop; the resulting .pte is copied to the Pi as-is.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from executorch.examples.models.llama.export_llama_lib import export_llama
from executorch.extension.llm.export.config.llm_config import LlmConfig
from omegaconf import OmegaConf

PATH_KEYS = ("base.checkpoint", "base.params", "base.tokenizer_path", "export.output_dir")


def main():
    overrides = [a for a in sys.argv[1:] if "=" in a]
    cfg_file = next((a for a in sys.argv[1:] if a.endswith((".yaml", ".yml"))), ROOT / "configs" / "llama_export.yaml")
    cli = OmegaConf.from_dotlist(overrides)
    meta = OmegaConf.select(cli, "base.metadata")
    if meta is not None and not isinstance(meta, str):  # dotlist parses the JSON string into a dict
        OmegaConf.update(cli, "base.metadata", json.dumps(OmegaConf.to_container(meta)), force_add=True)
    merged = OmegaConf.merge(OmegaConf.structured(LlmConfig), OmegaConf.load(cfg_file), cli)

    for key in PATH_KEYS:  # resolve project-relative paths
        val = OmegaConf.select(merged, key)
        if val and not Path(val).is_absolute():
            OmegaConf.update(merged, key, str(ROOT / val))
    ckpt = OmegaConf.select(merged, "base.checkpoint")
    if ckpt and not Path(ckpt).exists():
        sys.exit(f"checkpoint not found: {ckpt}\nSee docs/executorch_llama.md step 1 (download Llama 3.2 1B Instruct).")
    Path(merged.export.output_dir).mkdir(parents=True, exist_ok=True)
    # ExecuTorch uses an output_name ending in ".pte" verbatim (relative to the cwd), ignoring output_dir.
    out = Path(merged.export.output_dir) / merged.export.output_name
    OmegaConf.update(merged, "export.output_name", str(out))

    t0 = time.perf_counter()
    export_llama(OmegaConf.to_object(merged))
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
