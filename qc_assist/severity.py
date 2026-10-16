"""Turn an anomaly map into an operator-facing verdict: severity, size and location.

The cut-offs are starting heuristics — tune them per product with the
line's quality engineer and report the chosen values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anomaly import Prediction

SEVERITY_LEVELS = ("ok", "minor", "major", "critical")


@dataclass
class Verdict:
    severity: str
    score: float
    area_pct: float                          # % of the image flagged as anomalous
    bbox: tuple[int, int, int, int] | None   # x0, y0, x1, y1 in anomaly-map pixels
    region: str                              # e.g. "top-left", for the advisory prompt


def assess(pred: Prediction, minor_area_pct: float = 0.5, major_area_pct: float = 3.0,
           critical_score: float = 2.0) -> Verdict:
    if not pred.is_defect:
        return Verdict("ok", pred.score, 0.0, None, "none")

    mask = pred.anomaly_map > 1.0
    area_pct = 100.0 * float(mask.mean())
    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    region = _region_name((xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]))

    if area_pct >= major_area_pct or pred.score >= critical_score:
        severity = "critical" if area_pct >= major_area_pct and pred.score >= critical_score else "major"
    elif area_pct >= minor_area_pct:
        severity = "major" if pred.score >= 1.5 else "minor"
    else:
        severity = "minor"
    return Verdict(severity, pred.score, area_pct, bbox, region)


def _region_name(center: tuple[float, float]) -> str:
    cx, cy = center
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"
