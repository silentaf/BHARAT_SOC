"""QC-Assist: on-device visual inspection + Llama advisory for the Bharat AI-SoC Challenge (PS5)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep downloaded weights inside the project folder instead of ~/.cache.
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
