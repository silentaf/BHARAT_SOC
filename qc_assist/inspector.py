"""Inspection engine used by the dashboard: runs vision in a thread, keeps the latest
annotated frame and state, hands defects to the Llama worker, and supports
"teach a new board" (capture good frames -> fit a new model -> hot-swap)."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import PROJECT_ROOT
from .advisor import LlamaAdvisor
from .anomaly import DetectorConfig, PatchCore, resize_input
from .inspection_log import InspectionLog
from .io import frames, list_images, load_rgb, overlay
from .severity import assess


class Inspector:
    def __init__(self, cfg: dict, source: str, model_path: str | None = None,
                 max_fps: float | None = None, loop: bool = False):
        self.cfg = cfg
        self.product = cfg["product"]
        self.source, self.max_fps, self.loop = source, max_fps, loop
        self.model = PatchCore.load(model_path) if model_path else None
        self.log = InspectionLog(PROJECT_ROOT / cfg["log_db"])
        self.advisor = LlamaAdvisor.from_config(cfg["llama"], PROJECT_ROOT)
        self.lang = "en"

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending = None          # (row_id, Future) — one advisory in flight at a time
        self._teach = None            # {"n": int, "frames": [..]} while capturing
        self.latest_jpeg: bytes | None = None
        self.advisories: deque = deque(maxlen=20)
        self.state = {"status": "starting", "severity": None, "score": None,
                      "vision_ms": None, "fps": None, "image": None, "teach": None}

    # ---------- lifecycle ----------

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            for name, frame in frames(self.source):
                if self._stop.is_set():
                    return
                t0 = time.perf_counter()
                self._step(name, frame)
                if self.max_fps:
                    time.sleep(max(0.0, 1 / self.max_fps - (time.perf_counter() - t0)))
            if not self.loop:
                with self._lock:
                    self.state["status"] = "source finished"
                return

    # ---------- per frame ----------

    def _step(self, name: str, frame: np.ndarray) -> None:
        if self._teach is not None:
            self._collect_teach_frame(frame)
            return
        if self.model is None:
            self._publish(frame, None, "NO MODEL - press 'Teach new board'", status="no model")
            return

        t0 = time.perf_counter()
        pred = self.model.predict(frame)
        verdict = assess(pred, **self.cfg["severity"])
        vision_ms = (time.perf_counter() - t0) * 1000
        row = self.log.add(self.product, name, verdict, vision_ms)

        if verdict.severity != "ok" and self._pending is None:
            fut = self.advisor.advise_async(self.product, verdict, lang=self.lang,
                                            recent_defect_rate=self.log.recent_defect_rate(self.product))
            self._pending = (row, fut, verdict)
        self._harvest()

        label = f"{verdict.severity.upper()}  score {verdict.score:.2f}  {vision_ms:.0f} ms"
        self._publish(frame, pred.anomaly_map, label, status="inspecting", severity=verdict.severity,
                      score=round(verdict.score, 3), vision_ms=round(vision_ms, 1), image=Path(name).name)

    def _harvest(self) -> None:
        if self._pending and self._pending[1].done():
            row, fut, verdict = self._pending
            self._pending = None
            try:
                adv = fut.result()
            except Exception as e:  # runner crash must not stop inspection
                adv = type("R", (), {"text": f"advisor error: {e}", "source": "error"})()
            self.log.set_advisory(row, adv.text, adv.source)
            self.advisories.appendleft({"id": row, "time": time.strftime("%H:%M:%S"),
                                        "severity": verdict.severity, "region": verdict.region,
                                        "text": adv.text, "source": adv.source})

    def _publish(self, frame, amap, label, **state) -> None:
        vis = overlay(frame, amap if amap is not None else np.zeros(frame.shape[:2], np.float32), label)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        now = time.perf_counter()
        with self._lock:
            last = self.state.get("_t")
            self.state.update(state)
            self.state["fps"] = round(1 / (now - last), 1) if last and now > last else None
            self.state["_t"] = now
            if ok:
                self.latest_jpeg = jpg.tobytes()

    # ---------- teach a new board ----------

    def teach_from_camera(self, n: int = 20) -> None:
        self._teach = {"n": n, "frames": []}
        with self._lock:
            self.state["teach"] = f"capturing 0/{n} good boards"

    def teach_from_folder(self, folder: str) -> None:
        size = self.cfg["detector"]["image_size"]
        imgs = [resize_input(load_rgb(p), size) for p in list_images(folder)]
        if not imgs:
            raise ValueError(f"no images in {folder}")
        threading.Thread(target=self._fit, args=(imgs,), daemon=True).start()

    def _collect_teach_frame(self, frame: np.ndarray) -> None:
        t = self._teach
        t["frames"].append(resize_input(frame, self.cfg["detector"]["image_size"]))
        k, n = len(t["frames"]), t["n"]
        with self._lock:
            self.state["teach"] = f"capturing {k}/{n} good boards"
        self._publish(frame, None, f"TEACHING {k}/{n} - show GOOD boards", status="teaching")
        if k >= n:
            self._teach = None
            threading.Thread(target=self._fit, args=(t["frames"],), daemon=True).start()
        else:
            time.sleep(0.3)  # give the operator time to move/rotate the board between captures

    def _fit(self, imgs: list[np.ndarray]) -> None:
        with self._lock:
            self.state["teach"] = f"training on {len(imgs)} images..."
        t0 = time.perf_counter()
        det = dict(self.cfg["detector"], layers=tuple(self.cfg["detector"]["layers"]))
        model = PatchCore(DetectorConfig(**det))
        model.fit(imgs)
        path = PROJECT_ROOT / "models" / f"{self.product}_taught.pt"
        model.save(path)
        self.model = model  # atomic reference swap; next frame uses the new model
        with self._lock:
            self.state["teach"] = f"taught in {time.perf_counter() - t0:.1f}s ({len(imgs)} images) -> {path.name}"

    # ---------- read side ----------

    def snapshot(self) -> dict:
        with self._lock:
            s = {k: v for k, v in self.state.items() if not k.startswith("_")}
        s.update(product=self.product, lang=self.lang, llama=self.advisor.available, llm_stats=self.advisor.last_stats,
                 advisories=list(self.advisories))
        return s
