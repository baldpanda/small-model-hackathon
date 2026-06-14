from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_eval_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as input_file:
            return [normalize_record(row, index) for index, row in enumerate(csv.DictReader(input_file), start=1)]
    if suffix in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    raise ValueError(f"Unsupported eval input format: {path.suffix}. Use CSV or JSONL.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(normalize_record(record, line_number))
    return rows


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    transcript = str(record.get("transcript") or record.get("text") or "").strip()
    normalized: dict[str, Any] = {
        "id": str(record.get("id") or record.get("speech_id") or f"record-{index}"),
        "transcript": transcript,
        "stats": build_stats(record, transcript),
    }

    for key in ("category", "speech_type", "scenario_notes", "has_garble", "gold_feedback"):
        value = record.get(key)
        if value not in (None, ""):
            normalized[key] = value
    return normalized


def build_stats(record: dict[str, Any], transcript: str) -> dict[str, Any]:
    from speech_stats import build_transcript_stats

    duration_seconds = optional_float(record.get("duration_sec") or record.get("duration_seconds"))
    duration_mmss = str(record.get("duration_mmss") or "").strip() or None
    return build_transcript_stats(
        transcript,
        duration_seconds=duration_seconds,
        duration_mmss=duration_mmss,
    )


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
