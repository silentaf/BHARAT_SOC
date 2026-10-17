"""Defect-type classification on top of the anomaly detector.

The detector already describes every patch with a feature vector and finds its nearest
normal patch. For a defective board, the *residual* (feature minus nearest normal feature)
of its most anomalous patches says what changed: a missing part, melted plastic, a bent
pin... A small logistic regression on that residual names the defect type.

It needs labelled defect examples per product (VisA ships them; on a real line the
quality engineer labels a few), costs no extra network (works with the CPU, ONNX or
Hailo backends alike) and adds well under a millisecond per board.
Measured accuracy (cross-validated): scripts/train_types.py, docs/technical_report.md.
"""

from __future__ import annotations

import numpy as np

# How many of the most anomalous patches describe the defect (40x40 grid at 320 px).
TOP_PATCHES = 8

TYPE_HI = {
    "melt": "पिघलना", "missing": "गायब पुर्ज़ा", "bent": "मुड़ा पिन", "scratch": "खरोंच",
    "extra": "अतिरिक्त पुर्ज़ा", "dirt": "गंदगी", "damage": "टूट-फूट", "burnt": "जला हुआ",
    "wrong place": "गलत जगह लगा पुर्ज़ा",
}


def type_text(defect_type: str, lang: str = "en") -> str:
    return TYPE_HI.get(defect_type, defect_type) if lang == "hi" else defect_type


class DefectTypeClassifier:
    """Standardise + multinomial logistic regression, stored as plain arrays (no pickled
    sklearn objects in the model file); sklearn is only needed to fit."""

    def __init__(self, classes: list[str], mean: np.ndarray, scale: np.ndarray, coef: np.ndarray,
                 intercept: np.ndarray, min_confidence: float = 0.5):
        self.classes, self.mean, self.scale = list(classes), mean, scale
        self.coef, self.intercept, self.min_confidence = coef, intercept, min_confidence

    @classmethod
    def fit(cls, descriptors: np.ndarray, labels: list[str], C: float = 0.1) -> "DefectTypeClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(descriptors)
        lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(scaler.transform(descriptors), labels)
        coef, intercept = lr.coef_, lr.intercept_
        if len(lr.classes_) == 2:  # sklearn stores one row for two classes
            coef, intercept = np.vstack([-coef, coef]) / 2, np.array([-intercept[0], intercept[0]]) / 2
        # plain str: the model file is read with torch.load(weights_only=True), which refuses numpy scalars
        return cls([str(c) for c in lr.classes_], scaler.mean_, scaler.scale_, coef, intercept)

    def probabilities(self, descriptor: np.ndarray) -> np.ndarray:
        z = (np.asarray(descriptor, np.float64) - self.mean) / self.scale
        logits = self.coef @ z + self.intercept
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict(self, descriptor: np.ndarray) -> tuple[str | None, float]:
        """(type, confidence); type is None when the classifier is not confident enough."""
        p = self.probabilities(descriptor)
        i = int(p.argmax())
        return (self.classes[i] if p[i] >= self.min_confidence else None), float(p[i])

    def to_dict(self) -> dict:
        return {"classes": [str(c) for c in self.classes], "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept.tolist(),
                "min_confidence": self.min_confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "DefectTypeClassifier":
        return cls(d["classes"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]),
                   np.array(d["intercept"]), d.get("min_confidence", 0.5))
