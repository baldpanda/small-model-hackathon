from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    reviews = {str(row.get("id")): row for row in read_jsonl(args.reviews)}
    failures = check_expectations(records, reviews)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print(f"Validated {len(records)} expectation rows against {args.reviews}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated review JSONL against canary expectations.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def check_expectations(records: list[dict[str, Any]], reviews: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for record in records:
        record_id = str(record.get("id"))
        review_row = reviews.get(record_id)
        if review_row is None:
            failures.append(f"{record_id}: missing generated review")
            continue

        if review_row.get("scorecard_shape_valid") is False:
            failures.append(f"{record_id}: invalid scorecard shape: {review_row.get('scorecard_shape_issues')}")
        if review_row.get("quote_faithfulness_valid") is False:
            failures.append(f"{record_id}: invalid quote faithfulness: {review_row.get('quote_faithfulness_issues')}")

        review_text = normalize_text(str(review_row.get("review") or ""))
        expectations = record.get("expectations") or {}
        for group in expectations.get("must_include_any", []):
            alternatives = [normalize_text(str(item)) for item in group if str(item).strip()]
            if alternatives and not any(item in review_text for item in alternatives):
                failures.append(f"{record_id}: missing one of {group}")
        for phrase in expectations.get("must_include", []):
            normalized = normalize_text(str(phrase))
            if normalized and normalized not in review_text:
                failures.append(f"{record_id}: missing required phrase {phrase!r}")
        for phrase in expectations.get("must_not_include", []):
            normalized = normalize_text(str(phrase))
            if normalized and normalized in review_text:
                failures.append(f"{record_id}: contains forbidden phrase {phrase!r}")
    return failures


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


if __name__ == "__main__":
    main()
