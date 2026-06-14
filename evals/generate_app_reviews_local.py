from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("generate_app_reviews")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EVALS_DIR))
    from eval_data import read_eval_records

    records = read_eval_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    from review import PROMPT_VERSION, review_speech, scorecard_shape_issues

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = read_completed_ids(args.output) if args.resume else set()
    output_mode = "a" if args.resume else "w"
    if completed_ids:
        LOGGER.info("Resuming %s with %s completed IDs", args.output, len(completed_ids))

    with args.output.open(output_mode, encoding="utf-8") as output_file:
        for index, record in enumerate(records, start=1):
            transcript_id = str(record.get("id") or f"record-{index}")
            if transcript_id in completed_ids:
                LOGGER.info("Skipping %s (%s/%s): already complete", transcript_id, index, len(records))
                continue
            transcript = str(record.get("transcript") or "").strip()
            if not transcript:
                raise ValueError(f"{transcript_id} has no transcript")

            LOGGER.info("Generating app review for %s (%s/%s)", transcript_id, index, len(records))
            result = {
                "id": transcript_id,
                "review_model": "app.review.review_speech",
                "prompt_version": PROMPT_VERSION,
                "stats": record["stats"],
            }
            try:
                result["review"] = call_with_timeout(
                    lambda: review_speech(transcript, stats=record["stats"]),
                    timeout_seconds=args.per_item_timeout_seconds,
                )
                result["scorecard_shape_issues"] = scorecard_shape_issues(result["review"])
                result["scorecard_shape_valid"] = not result["scorecard_shape_issues"]
            except Exception as exc:  # noqa: BLE001 - eval rows should capture failures and continue.
                LOGGER.warning("Review failed for %s: %s", transcript_id, exc)
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
            if args.include_input:
                result["transcript"] = transcript
                for key in ("category", "speech_type", "scenario_notes", "has_garble", "gold_feedback"):
                    if key in record:
                        result[key] = record[key]
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MiniCPM reviews with the same review_speech path used by the HF app."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or JSONL with transcript records.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with generated app reviews.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Append missing IDs to an existing output JSONL.")
    parser.add_argument(
        "--per-item-timeout-seconds",
        type=int,
        default=240,
        help="Record an error and continue if one review generation exceeds this many seconds. Use 0 to disable.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Include transcripts and stats in the output JSONL. Keep outputs private if enabled.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            transcript_id = record.get("id")
            if transcript_id:
                completed_ids.add(str(transcript_id))
    return completed_ids


def call_with_timeout(function, timeout_seconds: int):
    if timeout_seconds <= 0:
        return function()

    def handle_timeout(signum, frame):
        raise TimeoutError(f"review generation exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(timeout_seconds)
    try:
        return function()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
