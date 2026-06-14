from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EVALS_DIR / "data" / "master_transcripts.csv"
DEFAULT_OUTPUT = EVALS_DIR / "private" / "master_transcript_stats.csv"
TEXT_FIELDS = {"transcript", "text", "gold_feedback"}
STAT_FIELDS = (
    "computed_word_count",
    "computed_filler_count",
    "computed_duration_seconds",
    "computed_duration_mmss",
    "computed_wpm",
    "computed_wpm_band",
    "computed_filler_per_min",
    "computed_filler_band",
    "computed_notable_fillers",
    "computed_filler_counts_json",
)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EVALS_DIR))

    from eval_data import build_stats

    rows = read_csv(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    output_rows = []
    for index, row in enumerate(rows, start=1):
        transcript = str(row.get("transcript") or row.get("text") or "").strip()
        if not transcript:
            raise ValueError(f"Row {index} has no transcript/text value")

        stats = build_stats(row, transcript)
        output_row = metadata_for_output(row, include_text=args.include_text)
        output_row.update(flatten_stats(stats))
        output_rows.append(output_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deterministic transcript stats to CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input transcript CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output stats CSV.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw transcript text in the output. Keep the output private if enabled.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def metadata_for_output(row: dict[str, Any], *, include_text: bool) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if not include_text and key in TEXT_FIELDS:
            continue
        output[key] = value
    return output


def flatten_stats(stats: dict[str, Any]) -> dict[str, Any]:
    notable_fillers = stats.get("notable_fillers") or []
    return {
        "computed_word_count": stats.get("word_count", ""),
        "computed_filler_count": stats.get("filler_count", ""),
        "computed_duration_seconds": stats.get("duration_seconds", ""),
        "computed_duration_mmss": stats.get("duration_mmss", ""),
        "computed_wpm": stats.get("wpm", ""),
        "computed_wpm_band": stats.get("wpm_band", ""),
        "computed_filler_per_min": stats.get("filler_per_min", ""),
        "computed_filler_band": stats.get("filler_band", ""),
        "computed_notable_fillers": format_notable_fillers(notable_fillers),
        "computed_filler_counts_json": json.dumps(stats.get("filler_counts", {}), sort_keys=True),
    }


def format_notable_fillers(notable_fillers: Any) -> str:
    if not isinstance(notable_fillers, list):
        return ""

    formatted = []
    for item in notable_fillers:
        if not isinstance(item, dict):
            continue
        filler = item.get("filler")
        count = item.get("count")
        if filler and count:
            formatted.append(f"{filler}:{count}")
    return ", ".join(formatted)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    for field in STAT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
