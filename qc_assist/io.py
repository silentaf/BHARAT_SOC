"""Image sources (folder or camera) and visual overlays."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def frames(source: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yields (name, RGB frame). `source` is an image folder, a camera index like "0", a
    snapshot URL ending in .jpg (e.g. the IP Webcam app on a phone: http://127.0.0.1:8080/shot.jpg)
    or a video stream URL."""
    if source.startswith(("http://", "https://")) and source.lower().endswith((".jpg", ".jpeg")):
        # One fresh snapshot per inspection: never a stale frame from a buffered stream.
        import urllib.request
        i = 0
        while True:
            with urllib.request.urlopen(source, timeout=5) as r:
                frame = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"no image from {source}")
            yield f"cam_{i:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    elif source.isdigit() or "://" in source:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prefer the newest frame
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {source}")
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"cam{source}_{i:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                i += 1
        finally:
            cap.release()
    else:
        for p in list_images(source):
            yield str(p), load_rgb(p)


def overlay(image_rgb: np.ndarray, anomaly_map: np.ndarray, label: str, max_width: int = 960) -> np.ndarray:
    """BGR display image: heat only where the score is at least half the threshold, a red outline
    around regions over the threshold, and a status label. Downscaled to max_width for streaming."""
    h, w = image_rgb.shape[:2]
    if w > max_width:
        h, w = int(h * max_width / w), max_width
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    amap = cv2.resize(anomaly_map, (w, h))
    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    heat = cv2.applyColorMap(np.clip(amap / 2.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.clip((amap - 0.5) / 0.5, 0, 1) * 0.55)[..., None]
    out = (base * (1 - alpha) + heat * alpha).astype(np.uint8)

    thick = max(2, w // 320)
    contours, _ = cv2.findContours((amap > 1.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), thick)

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1100)
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    color = (80, 200, 80) if label.startswith("OK") else (60, 60, 255)
    cv2.putText(out, label, (10, th + 10), font, scale, color, thick, cv2.LINE_AA)
    return out
