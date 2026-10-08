"""Results Store: persists every run's raw per-criterion scores (never just
the composite), matching the schema in the design doc's LLD, so a run can
be re-sliced later without re-running any judge calls.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from .aggregator import RowResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    module TEXT NOT NULL,
    started_at REAL NOT NULL,
    baseline_run_id TEXT
);
CREATE TABLE IF NOT EXISTS rows (
    row_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    input TEXT, golden TEXT, generated TEXT, context TEXT,
    composite REAL, passed INTEGER,
    PRIMARY KEY (run_id, row_id)
);
CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT NOT NULL,
    row_id TEXT NOT NULL,
    criterion TEXT NOT NULL,
    score REAL NOT NULL,
    rationale TEXT,
    judge_provider TEXT,
    judge_model TEXT
);
"""


class ResultsStore:
    def __init__(self, db_path: str | Path = "eval_results.db") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def start_run(self, module: str, baseline_run_id: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO runs (run_id, module, started_at, baseline_run_id) VALUES (?, ?, ?, ?)",
            (run_id, module, time.time(), baseline_run_id),
        )
        self._conn.commit()
        return run_id

    def save_results(self, run_id: str, results: list[RowResult]) -> None:
        for r in results:
            self._conn.execute(
                "INSERT INTO rows (row_id, run_id, input, golden, generated, context, composite, passed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (r.row.row_id, run_id, r.row.input, r.row.golden, r.row.generated, r.row.context,
                 r.composite, int(r.passed)),
            )
            for s in r.scores:
                self._conn.execute(
                    "INSERT INTO scores (run_id, row_id, criterion, score, rationale, judge_provider, judge_model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, r.row.row_id, s.criterion, s.score, s.rationale, s.judge_provider, s.judge_model),
                )
        self._conn.commit()

    def latest_run(self, module: str, before_run_id: str | None = None) -> str | None:
        query = "SELECT run_id FROM runs WHERE module = ?"
        params: list = [module]
        if before_run_id is not None:
            query += " AND run_id != ? ORDER BY started_at DESC LIMIT 1"
            params.append(before_run_id)
        else:
            query += " ORDER BY started_at DESC LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else None

    def avg_scores_for_run(self, run_id: str | None) -> dict[str, float]:
        if run_id is None:
            return {}
        rows = self._conn.execute(
            "SELECT criterion, AVG(score) FROM scores WHERE run_id = ? GROUP BY criterion", (run_id,)
        ).fetchall()
        return {criterion: avg for criterion, avg in rows}

    def close(self) -> None:
        self._conn.close()
