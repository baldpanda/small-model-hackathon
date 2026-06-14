from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_CSV = EVALS_DIR / "data" / "eval_transcripts.csv"


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EVALS_DIR))
    from prepare_sft_dataset import read_csv, transcript_hashes, validate_assistant_scorecard

    rows = read_jsonl(args.input)
    eval_hashes = transcript_hashes(read_csv(args.eval_csv), transcript_keys=("transcript", "text"))
    validate_rows(rows, eval_hashes=eval_hashes, validate_assistant_scorecard=validate_assistant_scorecard)
    print(f"Validated {len(rows)} SFT rows in {args.input}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MiniCPM5 SFT messages JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
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


def validate_rows(rows: list[dict[str, Any]], *, eval_hashes: set[str], validate_assistant_scorecard) -> None:
    seen_ids = set()
    seen_sources = set()
    for index, row in enumerate(rows, start=1):
        row_id = required_string(row, "id", index)
        source_id = required_string(row, "source_id", index)
        if row_id in seen_ids:
            raise ValueError(f"Duplicate row id {row_id}")
        if source_id in seen_sources:
            raise ValueError(f"Duplicate source_id {source_id}")
        seen_ids.add(row_id)
        seen_sources.add(source_id)

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{row_id} must have exactly 3 messages")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{row_id} messages must have system/user/assistant roles")

        user_content = str(messages[1].get("content") or "")
        if "Stats:" not in user_content or "Transcript:" not in user_content:
            raise ValueError(f"{row_id} user message must include Stats and Transcript sections")
        transcript = user_content.split("Transcript:", 1)[1].strip()
        if not transcript:
            raise ValueError(f"{row_id} transcript is empty")

        try:
            from prepare_sft_dataset import normalized_hash
        except ModuleNotFoundError:
            from evals.prepare_sft_dataset import normalized_hash

        if normalized_hash(transcript) in eval_hashes:
            raise ValueError(f"{row_id} overlaps final eval set by transcript hash")
        validate_assistant_scorecard(str(messages[2].get("content") or ""))


def required_string(row: dict[str, Any], key: str, index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Row {index} missing {key}")
    return value


if __name__ == "__main__":
    main()
