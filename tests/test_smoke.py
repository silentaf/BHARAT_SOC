"""End-to-end smoke test on synthetic textured 'boards' — no dataset needed.

Downloads ResNet-18 weights (~45 MB) into .cache/torch on first run.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_assist.advisor import LlamaAdvisor, build_prompt
from qc_assist.anomaly import DetectorConfig, PatchCore
from qc_assist.inspection_log import InspectionLog
from qc_assist.severity import assess


def board(rng, defect=False):
    img = np.full((224, 224, 3), (30, 110, 40), np.uint8)       # green solder mask
    img[::16, :, :] = (200, 170, 60)                            # copper traces
    img[:, ::16, :] = (200, 170, 60)
    img = np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    if defect:
        img[90:130, 100:150] = (10, 10, 10)                      # burn / missing component
    return img


def test_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    model = PatchCore(DetectorConfig(coreset_ratio=0.05))
    model.fit([board(rng) for _ in range(20)])

    good = model.predict(board(rng))
    bad = model.predict(board(rng, defect=True))
    assert bad.score > good.score
    assert bad.is_defect

    verdict = assess(bad)
    assert verdict.severity in ("minor", "major", "critical")
    assert verdict.region == "centre"

    path = tmp_path / "m.pt"
    model.save(path)
    assert abs(PatchCore.load(path).predict(board(rng, defect=True)).score - bad.score) < 0.2

    log = InspectionLog(tmp_path / "log.sqlite")
    row = log.add("pcb", "synthetic", verdict, 10.0)
    adv = LlamaAdvisor(None, None, None).advise("pcb", verdict)
    assert adv.source == "fallback"
    log.set_advisory(row, adv.text, adv.source)
    assert log.recent_defect_rate("pcb") == 1.0
    assert "<|start_header_id|>assistant" in build_prompt("pcb", verdict)
