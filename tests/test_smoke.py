"""End-to-end smoke test on synthetic textured 'boards' — no dataset needed.

Downloads ResNet-18 weights (~45 MB) into .cache/torch on first run.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_assist import advisor as adv_mod
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


def test_llama_output_checks():
    """The checks that decide whether a Llama answer is shown (see qc_assist/advisor.py)."""
    stats = adv_mod.EXAMPLE_POOR                          # rising trend, 4 critical defects
    facts = adv_mod.stats_facts(stats)
    assert "Shift rating: POOR" in facts and "RISING" in facts
    ok = "pcb2 had a poor shift: 31 of 290 boards were defective (10.7%). Most defects were in the bottom-right (14)."
    assert adv_mod.check_output(ok, facts, stats) is None
    assert "numbers" in adv_mod.check_output("31 of 300 boards were defective.", facts, stats)
    assert "locations" in adv_mod.check_output("Most defects were in the centre.", facts, stats)
    assert "rising" in adv_mod.check_output("The defect rate was stable.", facts, stats)
    assert "critical" in adv_mod.check_output("There were no critical defects.", facts, stats)
    assert "incomplete" in adv_mod.check_output("Most defects were in the", facts, stats)
    hi_facts = adv_mod.stats_facts(stats, "hi")
    assert "bottom-right (नीचे-दाएँ)" in hi_facts
    assert adv_mod.check_output("ज़्यादातर दोष नीचे-दाएँ थे (14)।", hi_facts, stats) is None
    assert "locations" in adv_mod.check_output("ज़्यादातर दोष ऊपर-दाएँ थे (14)।", hi_facts, stats)
    assert adv_mod.clean_output("Here is the note:\nAll good. Then the") == "All good."

    advisor = LlamaAdvisor(None, None, None)
    assert advisor.answer("Which operator was on shift?", stats).source == "template"
    assert advisor.report(stats, "hi").source == "fallback"
    assert "बढ़ रहा" in advisor.report(stats, "hi").text
