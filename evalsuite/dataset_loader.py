"""Reads the (golden, generated) pairs for a run -- the Dataset Loader from
the design doc's HLD. Source-agnostic on purpose: JSONL/CSV today, a
DB/object-store table later, the same EvalRow objects out either way.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from .models import EvalRow


def load_jsonl(path: str | Path) -> Iterator[EvalRow]:
    path = Path(path)
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield EvalRow(
                row_id=str(row.get("row_id", i)),
                module=row["module"],
                input=row["input"],
                golden=row["golden"],
                generated=row["generated"],
                context=row.get("context"),
            )


def load_csv(path: str | Path) -> Iterator[EvalRow]:
    path = Path(path)
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            yield EvalRow(
                row_id=str(row.get("row_id", i)),
                module=row["module"],
                input=row["input"],
                golden=row["golden"],
                generated=row["generated"],
                context=row.get("context") or None,
            )


def load_dataset(path: str | Path) -> list[EvalRow]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return list(load_jsonl(path))
    if path.suffix == ".csv":
        return list(load_csv(path))
    raise ValueError(f"unsupported dataset format: {path.suffix!r} (use .jsonl or .csv)")
