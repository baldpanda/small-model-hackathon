"""Sanitize a Codex JSONL trace for sharing.

This is a best-effort helper, not a privacy guarantee. Always manually review
the sanitized output before publishing it publicly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SENSITIVE_PATTERN_SPECS = (
    ("openai_key", r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_OPENAI_KEY]", 0),
    ("huggingface_token", r"hf_[A-Za-z0-9_-]{20,}", "[REDACTED_HF_TOKEN]", 0),
    ("github_fine_grained_pat", r"github_pat_[A-Za-z0-9_]{20,}", "[REDACTED_GITHUB_TOKEN]", 0),
    ("github_classic_token", r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]", 0),
    ("google_api_key", r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_GOOGLE_API_KEY]", 0),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[REDACTED_AWS_ACCESS_KEY]", 0),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{20,}", "[REDACTED_SLACK_TOKEN]", 0),
    (
        "private_key_block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        re.DOTALL,
    ),
    ("bearer_token", r"\bbearer\s+[A-Za-z0-9._~+/-]{20,}", "Bearer [REDACTED_BEARER_TOKEN]", re.IGNORECASE),
)

SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Z0-9_.-]*(?:api[_-]?key|token|secret|password|authorization)[A-Z0-9_.-]*\s*[:=]\s*)([\"']?)([^\"'\s,}]+)",
    re.IGNORECASE,
)

SESSION_META_STRIP_KEYS = {
    "base_instructions",
}


def build_replacements(workspace: Path, manual_terms: list[str]) -> list[tuple[str, re.Pattern[str], str]]:
    replacements: list[tuple[str, re.Pattern[str], str]] = []
    for name, pattern, replacement, flags in SENSITIVE_PATTERN_SPECS:
        replacements.append((name, re.compile(pattern, flags), replacement))

    paths = [workspace.resolve(), Path.home().resolve()]
    for path in paths:
        value = str(path)
        if value:
            replacement = "[WORKSPACE]" if path == workspace.resolve() else "[HOME]"
            replacements.append((f"path:{value}", re.compile(re.escape(value)), replacement))

    home_parent = Path.home().resolve().parent
    replacements.append(
        (
            "user_home_parent",
            re.compile(re.escape(str(home_parent)) + r"/[^/\"'\s]+"),
            str(home_parent) + "/[USER]",
        )
    )

    for index, term in enumerate(manual_terms, start=1):
        if term:
            replacements.append(
                (
                    f"manual_term_{index}",
                    re.compile(re.escape(term), re.IGNORECASE),
                    "[REDACTED_PERSONAL_DETAIL]",
                )
            )
    return replacements


def redact_string(value: str, replacements: list[tuple[str, re.Pattern[str], str]], stats: dict[str, int]) -> str:
    redacted = value
    for name, pattern, replacement in replacements:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            stats[name] = stats.get(name, 0) + count

    redacted, count = SENSITIVE_ASSIGNMENT_PATTERN.subn(r"\1\2[REDACTED_SECRET]", redacted)
    if count:
        stats["sensitive_assignment"] = stats.get("sensitive_assignment", 0) + count
    return redacted


def redact_value(value: Any, replacements: list[tuple[str, re.Pattern[str], str]], stats: dict[str, int]) -> Any:
    if isinstance(value, str):
        return redact_string(value, replacements, stats)
    if isinstance(value, list):
        return [redact_value(item, replacements, stats) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            redacted_key = redact_string(key, replacements, stats)
            redacted[redacted_key] = (
                "[REDACTED_SESSION_METADATA]"
                if key in SESSION_META_STRIP_KEYS
                else redact_value(item, replacements, stats)
            )
        return redacted
    return value


def should_drop_event(event: dict[str, Any], keep_reasoning: bool) -> bool:
    if keep_reasoning:
        return False
    payload = event.get("payload")
    if event.get("type") == "response_item" and isinstance(payload, dict):
        return payload.get("type") == "reasoning"
    return False


def sanitize_trace(
    input_path: Path,
    output_path: Path,
    workspace: Path,
    manual_terms: list[str],
    keep_reasoning: bool,
) -> dict[str, int]:
    replacements = build_replacements(workspace, manual_terms)
    stats: dict[str, int] = {
        "input_lines": 0,
        "output_lines": 0,
        "dropped_reasoning_lines": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stats["input_lines"] += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSONL: {exc}") from exc

            if isinstance(event, dict) and should_drop_event(event, keep_reasoning):
                stats["dropped_reasoning_lines"] += 1
                continue

            redacted_event = redact_value(event, replacements, stats)
            destination.write(json.dumps(redacted_event, sort_keys=False) + "\n")
            stats["output_lines"] += 1

    return stats


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".jsonl"):
        return input_path.with_name(input_path.name.removesuffix(".jsonl") + ".sanitized.jsonl")
    return input_path.with_name(input_path.name + ".sanitized.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize a Codex JSONL trace for sharing. This is best-effort only; "
            "manually review output before publishing."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the raw Codex JSONL trace.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the sanitized JSONL trace. Defaults to INPUT.sanitized.jsonl.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace path to redact. Defaults to the current directory.",
    )
    parser.add_argument(
        "--redact-term",
        action="append",
        default=[],
        help="Additional case-insensitive phrase to replace with [REDACTED_PERSONAL_DETAIL]. Repeat as needed.",
    )
    parser.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="Keep Codex reasoning trace items. By default they are dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (args.output.expanduser() if args.output else default_output_path(input_path)).resolve()
    workspace = args.workspace.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Output path must be different from input path.")

    stats = sanitize_trace(input_path, output_path, workspace, args.redact_term, args.keep_reasoning)

    print(f"Wrote {output_path}")
    print(f"Input lines: {stats['input_lines']}")
    print(f"Output lines: {stats['output_lines']}")
    print(f"Dropped reasoning lines: {stats['dropped_reasoning_lines']}")
    print("Redactions:")
    hidden_stats = {"input_lines", "output_lines", "dropped_reasoning_lines"}
    for name in sorted(key for key in stats if key not in hidden_stats):
        print(f"  {name}: {stats[name]}")


if __name__ == "__main__":
    main()
