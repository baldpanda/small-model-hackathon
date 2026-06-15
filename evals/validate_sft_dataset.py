from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import re
import sys
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
DEFAULT_EVAL_CSV = EVALS_DIR / "data" / "eval_transcripts.csv"


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EVALS_DIR))
    from prepare_sft_dataset import read_csv, transcript_hashes, validate_assistant_scorecard

    rows = read_jsonl(args.input)
    eval_hashes = transcript_hashes(read_csv(args.eval_csv), transcript_keys=("transcript", "text"))
    validate_rows(
        rows,
        eval_hashes=eval_hashes,
        validate_assistant_scorecard=validate_assistant_scorecard,
        check_quote_faithfulness=args.check_quote_faithfulness,
        check_distinct_fixes=args.check_distinct_fixes,
        max_fix_similarity=args.max_fix_similarity,
    )
    fix_counts = assistant_fix_count_distribution(rows)
    single_fix_rows = fix_counts.get(1, 0)
    if single_fix_rows < args.min_single_fix_rows:
        raise ValueError(
            f"{args.input} has {single_fix_rows} single-fix rows; expected at least {args.min_single_fix_rows}"
        )
    print(f"Validated {len(rows)} SFT rows in {args.input}")
    print("Assistant fix-count distribution: " + format_fix_counts(fix_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MiniCPM5 SFT messages JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument(
        "--min-single-fix-rows",
        type=int,
        default=0,
        help="Fail if fewer than this many rows contain exactly one middle fix/polish bullet.",
    )
    parser.add_argument(
        "--check-quote-faithfulness",
        action="store_true",
        help="Fail if assistant double-quoted spans are not found in the transcript after light normalization.",
    )
    parser.add_argument(
        "--check-distinct-fixes",
        action="store_true",
        help="Fail if a gold assistant output has duplicate or near-duplicate fix bullets.",
    )
    parser.add_argument(
        "--max-fix-similarity",
        type=float,
        default=0.86,
        help="Maximum allowed normalized SequenceMatcher similarity between two fix bullets.",
    )
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


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    eval_hashes: set[str],
    validate_assistant_scorecard,
    check_quote_faithfulness: bool = False,
    check_distinct_fixes: bool = False,
    max_fix_similarity: float = 0.86,
) -> None:
    if check_quote_faithfulness:
        from review import quote_faithfulness_issues

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
        assistant = str(messages[2].get("content") or "")
        validate_assistant_scorecard(assistant)
        if check_distinct_fixes:
            issues = fix_distinctness_issues(assistant, max_similarity=max_fix_similarity)
            if issues:
                formatted = "; ".join(issues[:3])
                raise ValueError(f"{row_id} assistant fix-distinctness failed: {formatted}")
        if check_quote_faithfulness:
            issues = quote_faithfulness_issues(assistant, transcript)
            if issues:
                formatted = "; ".join(issues[:3])
                raise ValueError(f"{row_id} assistant quote-faithfulness failed: {formatted}")


def assistant_fix_count_distribution(rows: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rows:
        messages = row.get("messages") or []
        assistant = str(messages[2].get("content") or "") if len(messages) >= 3 else ""
        counts[count_middle_bullets(assistant)] += 1
    return counts


def count_middle_bullets(assistant: str) -> int:
    labels = []
    for line in assistant.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        labels.append(stripped[2:].split(":", 1)[0].strip())
    if len(labels) < 2:
        return 0
    return len(labels[1:-1])


def fix_distinctness_issues(assistant: str, *, max_similarity: float = 0.86) -> list[str]:
    fixes = fix_bullet_contents(assistant)
    issues = []
    for left_index, left in enumerate(fixes):
        for right_index, right in enumerate(fixes[left_index + 1 :], start=left_index + 2):
            left_norm = normalize_fix_text(left)
            right_norm = normalize_fix_text(right)
            if not left_norm or not right_norm:
                continue
            similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
            if left_norm == right_norm or similarity >= max_similarity:
                issues.append(
                    f"Fix {left_index + 1} and Fix {right_index} are too similar "
                    f"({similarity:.2f}): {left!r} / {right!r}"
                )
    return issues


def fix_bullet_contents(assistant: str) -> list[str]:
    contents = []
    for line in assistant.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        label, content = stripped[2:].split(":", 1)
        if label.strip().startswith("Fix "):
            contents.append(content.strip())
    return contents


def normalize_fix_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def format_fix_counts(counts: Counter[int]) -> str:
    return ", ".join(f"{fix_count} fix(es): {counts[fix_count]}" for fix_count in sorted(counts)) or "none"


def required_string(row: dict[str, Any], key: str, index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Row {index} missing {key}")
    return value


if __name__ == "__main__":
    main()
