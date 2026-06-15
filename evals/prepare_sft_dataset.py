from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD_CSV = EVALS_DIR / "data" / "master_transcripts_gold.csv"
DEFAULT_STATS_CSV = EVALS_DIR / "private" / "master_transcript_stats.csv"
DEFAULT_EVAL_CSV = EVALS_DIR / "data" / "eval_transcripts.csv"
DEFAULT_OUTPUT_DIR = EVALS_DIR / "private"
DEFAULT_AUGMENT_JSONL = EVALS_DIR / "private" / "semantic_faithfulness_augments.jsonl"
TRAIN_OUTPUT_NAME = "sft_train_messages.jsonl"
VAL_OUTPUT_NAME = "sft_val_messages.jsonl"
SMOKE_OUTPUT_NAME = "sft_smoke_messages.jsonl"
SPLIT_REPORT_NAME = "sft_split_report.json"
VAL_FRACTION = 0.2
SEED = 42
SMOKE_SIZE = 8
ALLOWED_ASSISTANT_LABELS = {"Strength", "Fix 1", "Fix 2", "Fix 3", "Next run"}
LABEL_MAP = {
    "strength": "Strength",
    "fix": "Fix",
    "fix 1": "Fix",
    "fix 2": "Fix",
    "fix 3": "Fix",
    "polish": "Fix",
    "next run": "Next run",
    "next step": "Next run",
    "the one thing that matters": "Fix",
}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))

    records = prepare_records(
        gold_rows=read_unique_csv(args.gold_csv, "gold"),
        stats_rows=read_unique_csv(args.stats_csv, "stats"),
        eval_rows=read_csv(args.eval_csv) if args.eval_csv else [],
        augment_rows=read_augment_rows(args),
        clear_augment_repeat=args.clear_augment_repeat,
        seed=args.seed,
        val_fraction=args.val_fraction,
        smoke_size=args.smoke_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / TRAIN_OUTPUT_NAME, records["train"])
    write_jsonl(args.output_dir / VAL_OUTPUT_NAME, records["val"])
    write_jsonl(args.output_dir / SMOKE_OUTPUT_NAME, records["smoke"])
    write_json(args.output_dir / SPLIT_REPORT_NAME, build_split_report(records))

    print(
        "Wrote "
        f"{len(records['train'])} train, "
        f"{len(records['val'])} validation, "
        f"{len(records['smoke'])} smoke rows to {args.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MiniCPM5 SFT messages JSONL from gold transcript CSV.")
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--stats-csv", type=Path, default=DEFAULT_STATS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--augment-jsonl", type=Path, default=DEFAULT_AUGMENT_JSONL)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--clear-augment-repeat",
        action="store_true",
        help="Ignore repeat counts in augment rows so each targeted example appears once.",
    )
    parser.add_argument(
        "--exclude-augment-canaries",
        action="store_true",
        help="Drop augment rows marked as canaries or whose quality/type contains 'canary'.",
    )
    parser.add_argument("--exclude-augment-quality", action="append", default=[])
    parser.add_argument("--exclude-augment-type", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--smoke-size", type=int, default=SMOKE_SIZE)
    return parser.parse_args()


def prepare_records(
    *,
    gold_rows: dict[str, dict[str, Any]],
    stats_rows: dict[str, dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    augment_rows: list[dict[str, Any]] | None = None,
    clear_augment_repeat: bool = False,
    seed: int,
    val_fraction: float,
    smoke_size: int,
) -> dict[str, list[dict[str, Any]]]:
    if set(gold_rows) != set(stats_rows):
        missing_stats = sorted(set(gold_rows) - set(stats_rows), key=sort_key)[:10]
        missing_gold = sorted(set(stats_rows) - set(gold_rows), key=sort_key)[:10]
        raise ValueError(f"Gold/stats ID mismatch. Missing stats: {missing_stats}; missing gold: {missing_gold}")

    eval_hashes = transcript_hashes(eval_rows, transcript_keys=("transcript", "text"))
    rows = [build_sft_record(gold_rows[row_id], stats_rows[row_id]) for row_id in sorted(gold_rows, key=sort_key)]
    augment_records = [
        build_augmented_sft_record(row, index=index)
        for index, row in enumerate(expand_augment_rows(augment_rows or [], clear_repeat=clear_augment_repeat), start=1)
    ]
    assert_unique_source_ids(rows + augment_records)

    overlaps = [row["source_id"] for row in rows + augment_records if row["transcript_hash"] in eval_hashes]
    if overlaps:
        raise ValueError(f"Training rows overlap final eval transcripts by hash: {overlaps[:10]}")

    train_rows, val_rows = split_train_val(rows, seed=seed, val_fraction=val_fraction)
    train_rows = sorted(train_rows + augment_records, key=lambda item: sort_key(item["source_id"]))
    smoke_rows = select_smoke_rows(train_rows, smoke_size=smoke_size)
    return {
        "train": [strip_internal_fields(row, split="train") for row in train_rows],
        "val": [strip_internal_fields(row, split="val") for row in val_rows],
        "smoke": [strip_internal_fields(row, split="smoke") for row in smoke_rows],
    }


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def read_unique_csv(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{label} row {index} has no id")
        if row_id in by_id:
            raise ValueError(f"{label} CSV has duplicate id {row_id}")
        by_id[row_id] = row
    return by_id


def read_augment_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.no_augment:
        return []
    path = args.augment_jsonl
    if not path.exists():
        if path == DEFAULT_AUGMENT_JSONL:
            return []
        raise FileNotFoundError(path)
    return filter_augment_rows(
        read_jsonl(path, "augment"),
        exclude_canaries=args.exclude_augment_canaries,
        excluded_qualities=set(args.exclude_augment_quality),
        excluded_types=set(args.exclude_augment_type),
    )


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} JSONL {path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} JSONL {path}:{line_number} must contain objects")
            rows.append(row)
    return rows


def filter_augment_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_canaries: bool,
    excluded_qualities: set[str],
    excluded_types: set[str],
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        quality = str(row.get("quality") or "")
        augmentation_type = str(row.get("augmentation_type") or "")
        if quality in excluded_qualities or augmentation_type in excluded_types:
            continue
        if exclude_canaries and is_augment_canary(row):
            continue
        filtered.append(row)
    return filtered


def is_augment_canary(row: dict[str, Any]) -> bool:
    quality = str(row.get("quality") or "").lower()
    augmentation_type = str(row.get("augmentation_type") or "").lower()
    return parse_bool(row.get("canary", False)) or "canary" in quality or "canary" in augmentation_type


def expand_augment_rows(rows: list[dict[str, Any]], *, clear_repeat: bool = False) -> list[dict[str, Any]]:
    expanded = []
    for index, row in enumerate(rows, start=1):
        base_id = str(row.get("id") or f"augment-{index:03d}").strip()
        repeat_count = 1 if clear_repeat else optional_int(row.get("repeat")) or 1
        if repeat_count < 1:
            raise ValueError(f"Augment row {base_id} has invalid repeat={repeat_count}")
        for repeat_index in range(1, repeat_count + 1):
            repeated = dict(row)
            if repeat_count > 1:
                repeated["id"] = f"{base_id}-repeat-{repeat_index}"
                repeated["repeat_source_id"] = base_id
                repeated["repeat_index"] = repeat_index
                repeated["repeat_count"] = repeat_count
            expanded.append(repeated)
    return expanded


def build_sft_record(gold_row: dict[str, Any], stats_row: dict[str, Any]) -> dict[str, Any]:
    from review import _build_messages, clean_unfaithful_review_quotes

    source_id = str(gold_row["id"]).strip()
    transcript = required_text(gold_row, "text", source_id)
    assistant = normalize_gold_review(required_text(gold_row, "gold_review", source_id))
    assistant = clean_unfaithful_review_quotes(assistant, transcript)
    validate_assistant_scorecard(assistant)
    stats = build_prompt_stats(gold_row, stats_row)
    messages = _build_messages(transcript, stats)
    messages.append({"role": "assistant", "content": assistant})

    return {
        "id": f"train-{int(source_id):03d}" if source_id.isdigit() else f"train-{source_id}",
        "source_id": source_id,
        "metadata": build_metadata(gold_row, stats_row, stats),
        "messages": messages,
        "transcript_hash": normalized_hash(transcript),
    }


def build_augmented_sft_record(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    from review import _build_messages, clean_unfaithful_review_quotes
    from speech_stats import build_transcript_stats

    source_id = str(row.get("id") or f"augment-{index:03d}").strip()
    transcript = str(row.get("text") or row.get("transcript") or "").strip()
    if not transcript:
        raise ValueError(f"Augment row {source_id} has no text/transcript")

    duration_mmss = str(row.get("duration_mmss") or row.get("computed_duration_mmss") or "").strip() or None
    duration_seconds = optional_float(
        row.get("duration_seconds") or row.get("duration_sec") or row.get("computed_duration_seconds")
    )
    if duration_seconds is None and duration_mmss:
        duration_seconds = parse_duration_mmss(duration_mmss)

    stats = build_transcript_stats(
        transcript,
        duration_seconds=duration_seconds,
        duration_mmss=duration_mmss,
    )
    assistant = normalize_gold_review(required_text(row, "gold_review", source_id))
    assistant = clean_unfaithful_review_quotes(assistant, transcript)
    validate_assistant_scorecard(assistant)
    messages = _build_messages(transcript, stats)
    messages.append({"role": "assistant", "content": assistant})

    return {
        "id": f"train-{source_id}",
        "source_id": source_id,
        "metadata": build_augment_metadata(row, stats),
        "messages": messages,
        "transcript_hash": normalized_hash(transcript),
    }


def build_augment_metadata(row: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "type": row.get("type", ""),
        "quality": row.get("quality", ""),
        "variant": row.get("variant", ""),
        "scenario_family": first_present_any((row,), ("scenario_family", "family_id", "family")),
        "garble": row.get("garble", ""),
        "speaker_role": row.get("speaker_role", ""),
        "augmentation": True,
        "augmentation_type": row.get("augmentation_type") or "semantic_faithfulness",
        "canary": parse_bool(row.get("canary", True)),
        "repeat_source_id": row.get("repeat_source_id", ""),
        "repeat_index": optional_int(row.get("repeat_index")),
        "repeat_count": optional_int(row.get("repeat_count")),
        "stats": stats,
        "filler_counts": stats.get("filler_counts", {}),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def required_text(row: dict[str, Any], key: str, source_id: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Row {source_id} has no {key}")
    return value


def build_prompt_stats(gold_row: dict[str, Any], stats_row: dict[str, Any]) -> dict[str, Any]:
    duration_mmss = first_present(gold_row, stats_row, "computed_duration_mmss")
    duration_seconds = optional_float(first_present(gold_row, stats_row, "computed_duration_seconds"))
    if duration_seconds is None and duration_mmss:
        duration_seconds = parse_duration_mmss(duration_mmss)

    return {
        "duration_seconds": duration_seconds,
        "duration_mmss": duration_mmss or None,
        "word_count": optional_int(stats_row.get("computed_word_count")),
        "wpm": optional_float(first_present(gold_row, stats_row, "computed_wpm")),
        "wpm_band": first_present(gold_row, stats_row, "computed_wpm_band") or None,
        "filler_count": optional_int(stats_row.get("computed_filler_count")),
        "filler_per_min": optional_float(first_present(gold_row, stats_row, "computed_filler_per_min")),
        "filler_band": first_present(gold_row, stats_row, "computed_filler_band") or None,
        "notable_fillers": parse_notable_fillers(stats_row.get("computed_notable_fillers")),
    }


def build_metadata(gold_row: dict[str, Any], stats_row: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "type": gold_row.get("type", ""),
        "quality": gold_row.get("quality", ""),
        "variant": gold_row.get("variant", ""),
        "scenario_family": first_present_any(
            (gold_row, stats_row),
            ("scenario_family", "family_id", "family"),
        ),
        "garble": gold_row.get("garble", ""),
        "length": stats_row.get("length", ""),
        "approx_words": optional_int(stats_row.get("approx_words")),
        "disfluency": stats_row.get("disfluency", ""),
        "speaker_role": stats_row.get("speaker_role", ""),
        "stats": stats,
        "filler_counts": parse_filler_counts(stats_row.get("computed_filler_counts_json")),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def normalize_gold_review(review: str) -> str:
    lines = [line.strip() for line in review.splitlines() if line.strip()]
    normalized_lines = []
    fix_count = 0
    for line in lines:
        body = strip_bullet(line)
        if ":" not in body:
            raise ValueError(f"Gold review line has no label: {body[:40]}")
        raw_label, content = body.split(":", 1)
        label = LABEL_MAP.get(raw_label.strip().lower())
        if label is None:
            raise ValueError(f"Unsupported gold review label: {raw_label}")
        if label == "Fix":
            fix_count += 1
            if fix_count > 3:
                raise ValueError("Gold review has more than 3 fix bullets")
            label = f"Fix {fix_count}"
        content = content.strip()
        if not content:
            raise ValueError(f"Gold review label has no content: {raw_label}")
        normalized_lines.append(f"- {label}: {content}")

    validate_assistant_scorecard("\n".join(normalized_lines))
    return "\n".join(normalized_lines)


def strip_bullet(line: str) -> str:
    if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
        return line[2:].strip()
    return line


def split_train_val(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row["metadata"]
        groups[(metadata.get("type", ""), metadata.get("quality", ""))].append(row)

    train_rows = []
    val_rows = []
    for key in sorted(groups):
        group_rows = sorted(groups[key], key=lambda item: sort_key(item["source_id"]))
        val_count = max(1, round(len(group_rows) * val_fraction)) if len(group_rows) > 1 else 0
        family_groups = list(group_by_scenario_family(group_rows).values())
        rng.shuffle(family_groups)

        group_val_rows = []
        group_train_rows = []
        for family_rows in family_groups:
            if len(group_val_rows) < val_count:
                group_val_rows.extend(family_rows)
            else:
                group_train_rows.extend(family_rows)
        val_rows.extend(group_val_rows)
        train_rows.extend(group_train_rows)

    return sorted(train_rows, key=lambda item: sort_key(item["source_id"])), sorted(
        val_rows, key=lambda item: sort_key(item["source_id"])
    )


def group_by_scenario_family(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[scenario_family_key(row)].append(row)
    return groups


def scenario_family_key(row: dict[str, Any]) -> str:
    family = str(row["metadata"].get("scenario_family") or "").strip()
    if family:
        return f"family:{family}"
    return f"source:{row['source_id']}"


def select_smoke_rows(train_rows: list[dict[str, Any]], smoke_size: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids = set()
    missing_types = {row["metadata"].get("type", "") for row in train_rows}
    missing_qualities = {row["metadata"].get("quality", "") for row in train_rows}

    candidates = sorted(train_rows, key=lambda item: sort_key(item["source_id"]))
    for row in candidates:
        if len(selected) >= smoke_size:
            break
        if is_smoke_canary(row):
            selected.append(row)
            selected_ids.add(row["source_id"])
            missing_types.discard(row["metadata"].get("type", ""))
            missing_qualities.discard(row["metadata"].get("quality", ""))

    while (missing_types or missing_qualities) and len(selected) < smoke_size:
        available_rows = [row for row in candidates if row["source_id"] not in selected_ids]
        if not available_rows:
            break
        best = max(
            available_rows,
            key=lambda row: (
                int(row["metadata"].get("type", "") in missing_types)
                + int(row["metadata"].get("quality", "") in missing_qualities),
                -sort_key(row["source_id"])[0],
                row["source_id"],
            ),
        )
        selected.append(best)
        selected_ids.add(best["source_id"])
        missing_types.discard(best["metadata"].get("type", ""))
        missing_qualities.discard(best["metadata"].get("quality", ""))

    for row in candidates:
        if len(selected) >= smoke_size:
            break
        if row["source_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["source_id"])

    if len(selected) != smoke_size:
        raise ValueError(f"Could only select {len(selected)} smoke rows, expected {smoke_size}")
    return sorted(selected, key=lambda item: sort_key(item["source_id"]))


def is_smoke_canary(row: dict[str, Any]) -> bool:
    metadata = row["metadata"]
    return bool(metadata.get("canary")) or metadata.get("augmentation_type") == "semantic_faithfulness"


def strip_internal_fields(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    public_row = {key: value for key, value in row.items() if key != "transcript_hash"}
    public_row["split"] = split
    public_row["metadata"] = {**public_row["metadata"], "split": split}
    return public_row


def build_split_report(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "counts": {split: len(rows) for split, rows in records.items()},
        "by_type": {split: count_metadata(rows, "type") for split, rows in records.items()},
        "by_quality": {split: count_metadata(rows, "quality") for split, rows in records.items()},
        "by_augmentation_type": {
            split: count_metadata(rows, "augmentation_type") for split, rows in records.items()
        },
        "by_scenario_family": {
            split: count_metadata(rows, "scenario_family") for split, rows in records.items()
        },
        "source_ids": {split: [row["source_id"] for row in rows] for split, rows in records.items()},
    }


def count_metadata(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row["metadata"].get(key, "")) for row in rows))


def validate_assistant_scorecard(text: str) -> None:
    if any(marker in text.lower() for marker in ("<think", "</think", "reasoning trace", "chain-of-thought")):
        raise ValueError("Assistant output contains thinking trace language")
    old_headers = {"What worked", "What to sharpen", "Try this next time", "Bottom line"}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line in old_headers for line in lines):
        raise ValueError("Assistant output contains old section headers")
    bullets = [line for line in lines if line.startswith("- ")]
    if len(bullets) != len(lines):
        raise ValueError("Assistant output must contain only hyphen bullets")
    if not 2 < len(bullets) <= 5:
        raise ValueError(f"Assistant output must have 3-5 bullets, found {len(bullets)}")
    labels = [line[2:].split(":", 1)[0].strip() if ":" in line else "" for line in bullets]
    if labels[0] != "Strength":
        raise ValueError("Assistant output must start with Strength")
    if labels[-1] != "Next run":
        raise ValueError("Assistant output must end with Next run")
    if any(label not in ALLOWED_ASSISTANT_LABELS for label in labels):
        raise ValueError(f"Assistant output has unsupported labels: {labels}")


def transcript_hashes(rows: list[dict[str, Any]], *, transcript_keys: tuple[str, ...]) -> set[str]:
    hashes = set()
    for row in rows:
        for key in transcript_keys:
            transcript = str(row.get(key) or "").strip()
            if transcript:
                hashes.add(normalized_hash(transcript))
                break
    return hashes


def normalized_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_notable_fillers(value: Any) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    fillers = []
    for item in text.split(","):
        if ":" not in item:
            continue
        filler, count = item.split(":", 1)
        count_int = optional_int(count.strip())
        if filler.strip() and count_int:
            fillers.append({"filler": filler.strip(), "count": count_int})
    return fillers


def parse_filler_counts(value: Any) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("computed_filler_counts_json must be a JSON object")
    return {str(key): int(count) for key, count in parsed.items()}


def parse_duration_mmss(value: str) -> float | None:
    parts = value.split(":")
    if len(parts) != 2:
        return None
    minutes, seconds = parts
    try:
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)


def assert_unique_source_ids(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["source_id"] for row in rows)
    duplicates = sorted(source_id for source_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate SFT source IDs: {duplicates[:10]}")


def first_present(*rows_and_key: Any) -> str:
    *rows, key = rows_and_key
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def first_present_any(rows: tuple[dict[str, Any], ...], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = first_present(*rows, key)
        if value:
            return value
    return ""


def optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(float(text))


def optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    if text.isdigit():
        return int(text), text
    return 10**9, text


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
