from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
DEFAULT_EVAL_INPUT = EVALS_DIR / "data" / "eval_transcripts.csv"
DEFAULT_SFT_INPUTS = [
    EVALS_DIR / "private" / "sft_train_messages.jsonl",
    EVALS_DIR / "private" / "sft_val_messages.jsonl",
]
CORE_STATS_LABELS = ("Duration", "Word count", "Pace", "Fillers")
OPTIONAL_STATS_LABELS = ("Notable fillers",)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EVALS_DIR))

    from eval_data import read_eval_records
    from review import format_review_stats

    eval_blocks = [
        {
            "id": record["id"],
            "stats_block": format_review_stats(record["stats"]),
        }
        for record in read_eval_records(args.eval_input)
    ]
    sft_blocks = []
    for path in args.sft_input:
        sft_blocks.extend(read_sft_stats_blocks(path))

    eval_report = build_schema_report(eval_blocks)
    sft_report = build_schema_report(sft_blocks)
    print_report("held-out eval", eval_report)
    print_report("SFT", sft_report)

    failures = []
    failures.extend(missing_core_failures("held-out eval", eval_report))
    failures.extend(missing_core_failures("SFT", sft_report))
    if eval_report["core_schema_counts"] != sft_report["core_schema_counts"] and args.require_matching_distribution:
        failures.append("held-out eval and SFT core schema distributions differ")
    if set(eval_report["core_schema_counts"]) != set(sft_report["core_schema_counts"]):
        failures.append("held-out eval and SFT use different core stats schemas")

    if failures:
        raise SystemExit("Stats schema consistency check failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check held-out eval and SFT stats-block schema consistency.")
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL_INPUT)
    parser.add_argument("--sft-input", type=Path, action="append", default=None)
    parser.add_argument(
        "--require-matching-distribution",
        action="store_true",
        help="Also require the same count distribution for each core schema, not just the same schema set.",
    )
    args = parser.parse_args()
    if args.sft_input is None:
        args.sft_input = DEFAULT_SFT_INPUTS
    return args


def read_sft_stats_blocks(path: Path) -> list[dict[str, str]]:
    blocks = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row_id = str(row.get("id") or f"{path.name}:{line_number}")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{row_id} has no user message")
            blocks.append({"id": row_id, "stats_block": extract_stats_block(str(messages[1].get("content") or ""))})
    return blocks


def extract_stats_block(user_content: str) -> str:
    if "Stats:" not in user_content:
        raise ValueError("User message has no Stats section")
    after_stats = user_content.split("Stats:", 1)[1]
    if "\n\n" in after_stats:
        return after_stats.split("\n\n", 1)[0].strip()
    return after_stats.strip()


def build_schema_report(blocks: list[dict[str, str]]) -> dict[str, Any]:
    core_schema_counts: Counter[tuple[str, ...]] = Counter()
    full_schema_counts: Counter[tuple[str, ...]] = Counter()
    missing_core: dict[str, list[str]] = {}
    for block in blocks:
        labels = stats_labels(block["stats_block"])
        core_labels = tuple(label for label in labels if label in CORE_STATS_LABELS)
        full_schema_counts[tuple(labels)] += 1
        core_schema_counts[core_labels] += 1
        missing = [label for label in CORE_STATS_LABELS if label not in labels]
        if missing:
            missing_core[block["id"]] = missing
    return {
        "count": len(blocks),
        "core_schema_counts": dict(core_schema_counts),
        "full_schema_counts": dict(full_schema_counts),
        "missing_core": missing_core,
    }


def stats_labels(stats_block: str) -> list[str]:
    labels = []
    for line in stats_block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        labels.append(stripped[2:].split(":", 1)[0].strip())
    return labels


def missing_core_failures(label: str, report: dict[str, Any]) -> list[str]:
    missing_core = report["missing_core"]
    if not missing_core:
        return []
    examples = list(missing_core.items())[:5]
    formatted = ", ".join(f"{row_id} missing {missing}" for row_id, missing in examples)
    return [f"{label} rows missing core stats labels: {formatted}"]


def print_report(label: str, report: dict[str, Any]) -> None:
    print(f"{label}: {report['count']} rows")
    print(f"{label} core schemas:")
    for schema, count in sorted(report["core_schema_counts"].items(), key=lambda item: (item[0], item[1])):
        print(f"  {count} x {', '.join(schema)}")
    notable_count = sum(
        count
        for schema, count in report["full_schema_counts"].items()
        if any(label in OPTIONAL_STATS_LABELS for label in schema)
    )
    print(f"{label} rows with notable fillers: {notable_count}")


if __name__ == "__main__":
    main()
