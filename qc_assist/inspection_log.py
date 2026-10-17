"""SQLite log of every inspection — the source for shift reports and the technical report's numbers."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .severity import Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    product TEXT NOT NULL,
    source TEXT,
    severity TEXT NOT NULL,
    score REAL NOT NULL,
    area_pct REAL NOT NULL,
    bbox TEXT,
    region TEXT,
    vision_ms REAL,
    advisory TEXT,
    advisory_source TEXT,
    defect_type TEXT
);
"""


class InspectionLog:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(inspections)")}
        if "defect_type" not in cols:  # logs written before defect types existed
            self.db.execute("ALTER TABLE inspections ADD COLUMN defect_type TEXT")

    def add(self, product: str, source: str, verdict: Verdict, vision_ms: float) -> int:
        cur = self.db.execute(
            "INSERT INTO inspections (ts, product, source, severity, score, area_pct, bbox, region, vision_ms,"
            " defect_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), product, source, verdict.severity, verdict.score, verdict.area_pct,
             json.dumps(verdict.bbox) if verdict.bbox else None, verdict.region, vision_ms, verdict.defect_type),
        )
        self.db.commit()
        return cur.lastrowid

    def set_advisory(self, row_id: int, text: str, source: str) -> None:
        self.db.execute("UPDATE inspections SET advisory = ?, advisory_source = ? WHERE id = ?", (text, source, row_id))
        self.db.commit()

    def summary(self, product: str, since_ts: float) -> dict:
        """Deterministic shift statistics. The LLM only phrases these facts, never computes them."""
        q = "FROM inspections WHERE product = ? AND ts >= ?"
        total = self.db.execute(f"SELECT COUNT(*) {q}", (product, since_ts)).fetchone()[0]
        by_sev = dict(self.db.execute(f"SELECT severity, COUNT(*) {q} GROUP BY severity", (product, since_ts)))
        regions = self.db.execute(
            f"SELECT region, COUNT(*) AS n {q} AND severity != 'ok' GROUP BY region ORDER BY n DESC LIMIT 3",
            (product, since_ts)).fetchall()
        hourly = self.db.execute(
            f"SELECT strftime('%H:00', ts, 'unixepoch', 'localtime') AS h, COUNT(*), SUM(severity != 'ok') "
            f"{q} GROUP BY h ORDER BY h", (product, since_ts)).fetchall()
        types = self.db.execute(
            f"SELECT defect_type, COUNT(*) AS n {q} AND severity != 'ok' AND defect_type IS NOT NULL "
            f"GROUP BY defect_type ORDER BY n DESC", (product, since_ts)).fetchall()
        avg_ms = self.db.execute(f"SELECT AVG(vision_ms) {q}", (product, since_ts)).fetchone()[0]
        defects = total - by_sev.get("ok", 0)
        return {
            "product": product,
            "inspected": total,
            "defective": defects,
            "defect_rate_pct": round(100 * defects / total, 1) if total else 0.0,
            "by_severity": {s: by_sev.get(s, 0) for s in ("ok", "minor", "major", "critical")},
            "top_defect_regions": [{"region": r, "count": n} for r, n in regions],
            "defect_types": [{"type": t, "count": n} for t, n in types],
            "hourly": [{"hour": h, "inspected": n, "defective": d} for h, n, d in hourly],
            "avg_vision_ms": round(avg_ms, 1) if avg_ms else None,
        }

    def recent(self, product: str, n: int = 20) -> list[dict]:
        cur = self.db.execute(
            "SELECT id, ts, source, severity, score, area_pct, region, defect_type, vision_ms, advisory, advisory_source "
            "FROM inspections WHERE product = ? ORDER BY id DESC LIMIT ?", (product, n))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def recent_defect_rate(self, product: str, n: int = 50) -> float | None:
        rows = self.db.execute(
            "SELECT severity FROM inspections WHERE product = ? ORDER BY id DESC LIMIT ?", (product, n)
        ).fetchall()
        if not rows:
            return None
        return sum(r[0] != "ok" for r in rows) / len(rows)
