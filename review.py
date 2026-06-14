from __future__ import annotations

import difflib
import logging
import os
import re
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


MODEL_ID = "openbmb/MiniCPM5-1B"
ADAPTER_ID_ENV = "REVIEW_ADAPTER_ID"
PROMPT_LOG_ENV = "REVIEW_LOG_PROMPT"
PROMPT_VERSION = "stats_variable_fix_scorecard_v4"
MAX_REVIEW_TOKENS = 260
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 0
PROMPTS_DIR = Path(__file__).with_name("prompts")
FIX_LABELS = ("Fix 1", "Fix 2", "Fix 3")
SCORECARD_LABELS = ("Strength", *FIX_LABELS, "Next run")
MIN_FIX_COUNT = 1
MAX_FIX_COUNT = len(FIX_LABELS)
ALLOWED_SCORECARD_LABELS = {label.lower() for label in SCORECARD_LABELS}
GENERIC_FIX_LABELS = {"fix", "polish"}

LOGGER = logging.getLogger(__name__)
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
DOUBLE_QUOTE_PATTERN = re.compile(r'"([^"\n]{3,240})"')


def _get_hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    return None


def _model_kwargs() -> dict[str, str]:
    token = _get_hugging_face_token()
    if token:
        return {"token": token}
    return {}


def _get_review_adapter_id() -> str | None:
    adapter_id = os.getenv(ADAPTER_ID_ENV, "").strip()
    return adapter_id or None


def _should_log_review_prompt() -> bool:
    value = os.getenv(PROMPT_LOG_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def _render_prompt(template_name: str, **context: Any) -> str:
    template = _prompt_environment().get_template(template_name)
    return template.render(**context).strip()


@lru_cache(maxsize=1)
def _load_review_stack() -> tuple[object, object, object]:
    load_started_at = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Review dependencies are missing. Install the project dependencies before running phase 4."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **_model_kwargs())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        **_model_kwargs(),
    )
    adapter_id = _get_review_adapter_id()
    if adapter_id:
        model = _load_review_adapter(model, adapter_id)
    runtime_device = _place_model_for_runtime(torch, model)
    model.eval()
    LOGGER.info(
        "Loaded review stack on %s with %s in %.1fs",
        runtime_device,
        _review_model_label(adapter_id),
        time.perf_counter() - load_started_at,
    )
    return torch, tokenizer, model


def _load_review_adapter(model: object, adapter_id: str) -> object:
    adapter_started_at = time.perf_counter()
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            f"{ADAPTER_ID_ENV} is set to {adapter_id!r}, but the 'peft' package is not installed."
        ) from exc

    try:
        adapted_model = PeftModel.from_pretrained(model, adapter_id, **_model_kwargs())
    except Exception as exc:
        raise RuntimeError(f"Failed to load review adapter {adapter_id!r}.") from exc

    LOGGER.info("Loaded review adapter %s in %.1fs", adapter_id, time.perf_counter() - adapter_started_at)
    return adapted_model


def _review_model_label(adapter_id: str | None) -> str:
    if adapter_id:
        return f"{MODEL_ID} + {adapter_id}"
    return MODEL_ID


def _place_model_for_runtime(torch: object, model: object) -> str:
    if torch.cuda.is_available():
        model.to("cuda")
        return "cuda"

    try:
        model.to("cuda")
    except (AssertionError, RuntimeError) as exc:
        LOGGER.info("CUDA startup placement unavailable; using CPU: %s", exc)
        model.to("cpu")
        return "cpu"

    return "cuda"


def _model_device(model: object) -> object:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def format_review_stats(stats: Mapping[str, Any] | None) -> str:
    if not stats:
        return "No stats supplied."

    lines: list[str] = []
    if stats.get("duration_mmss") and stats.get("duration_seconds") is not None:
        lines.append(
            f"- Duration: {stats['duration_mmss']} ({_format_decimal(stats['duration_seconds'])} seconds)"
        )
    elif stats.get("duration_seconds") is not None:
        lines.append(f"- Duration: {_format_decimal(stats['duration_seconds'])} seconds")

    if stats.get("word_count") is not None:
        lines.append(f"- Word count: {stats['word_count']}")

    if stats.get("wpm") is not None:
        pace = f"- Pace: {_format_decimal(stats['wpm'])} wpm"
        if stats.get("wpm_band"):
            pace += f" ({stats['wpm_band']})"
        lines.append(pace)

    if stats.get("filler_count") is not None:
        filler = f"- Fillers: {stats['filler_count']} total"
        if stats.get("filler_per_min") is not None:
            filler += f", {_format_decimal(stats['filler_per_min'])} per minute"
        if stats.get("filler_band"):
            filler += f" ({stats['filler_band']})"
        lines.append(filler)

    notable_fillers = stats.get("notable_fillers")
    if isinstance(notable_fillers, list) and notable_fillers:
        formatted = []
        for item in notable_fillers[:5]:
            if isinstance(item, Mapping) and item.get("filler") and item.get("count") is not None:
                formatted.append(f"{item['filler']} {item['count']}")
        if formatted:
            lines.append(f"- Notable fillers: {', '.join(formatted)}")

    return "\n".join(lines) if lines else "No stats supplied."


def _format_decimal(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.1f}"


def expected_scorecard_labels(
    transcript: str,
    stats: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    _ = transcript, stats
    return SCORECARD_LABELS


def clean_review_output(
    review: str,
    expected_labels: tuple[str, ...] = SCORECARD_LABELS,
) -> str:
    _ = expected_labels
    review = _strip_thinking_trace(review)
    lines = [line.strip() for line in review.strip().splitlines() if line.strip()]
    bullet_lines = [_normalize_scorecard_line(line) for line in lines]
    bullet_lines = [line for line in bullet_lines if line.startswith("- ")]
    if not bullet_lines:
        return review.strip()

    strength = ""
    fixes: list[str] = []
    next_run = ""

    for line in bullet_lines:
        label, content = _parse_scorecard_bullet(line)
        if label not in ALLOWED_SCORECARD_LABELS and label not in GENERIC_FIX_LABELS:
            continue
        if label == "strength":
            if strength:
                continue
            strength = content
            continue
        if not strength:
            continue
        if label in {"fix", "polish", "fix 1", "fix 2", "fix 3"}:
            if len(fixes) < MAX_FIX_COUNT:
                fixes.append(content)
            continue
        if label == "next run":
            next_run = content
            break

    if strength and len(fixes) >= MIN_FIX_COUNT and next_run:
        return "\n".join(_format_scorecard_lines(strength, fixes, next_run))
    return review.strip()


def _format_scorecard_lines(strength: str, fixes: list[str], next_run: str) -> list[str]:
    fixes = _dedupe_fixes(strength, fixes)
    lines = [f"- Strength: {strength}"]
    for index, fix in enumerate(fixes[:MAX_FIX_COUNT], start=1):
        lines.append(f"- Fix {index}: {fix}")
    lines.append(f"- Next run: {next_run}")
    return lines


def _dedupe_fixes(strength: str, fixes: list[str]) -> list[str]:
    selected: list[str] = []
    for fix in fixes:
        if not fix:
            continue
        if any(_feedback_items_too_similar(fix, existing) for existing in [strength, *selected]):
            continue
        selected.append(fix)

    return selected or fixes[:MIN_FIX_COUNT]


def _feedback_items_too_similar(left: str, right: str) -> bool:
    left_normalized = _normalize_feedback_for_similarity(left)
    right_normalized = _normalize_feedback_for_similarity(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True

    sequence_ratio = difflib.SequenceMatcher(None, left_normalized, right_normalized).ratio()
    if sequence_ratio >= 0.86:
        return True

    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    if min(len(left_tokens), len(right_tokens)) < 5:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.78


def _normalize_feedback_for_similarity(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _strip_thinking_trace(review: str) -> str:
    without_blocks = THINK_BLOCK_PATTERN.sub("", review)
    return THINK_TAG_PATTERN.sub("", without_blocks).strip()


def _normalize_scorecard_line(line: str) -> str:
    if line.startswith("- "):
        return line
    label = _plain_line_label(line)
    if label in ALLOWED_SCORECARD_LABELS or label in GENERIC_FIX_LABELS:
        return f"- {line}"
    return line


def _plain_line_label(line: str) -> str:
    if ":" not in line:
        return ""
    return line.split(":", 1)[0].strip().lower()


def is_valid_scorecard_shape(
    review: str,
    expected_labels: tuple[str, ...] = SCORECARD_LABELS,
) -> bool:
    return not scorecard_shape_issues(review, expected_labels=expected_labels)


def scorecard_shape_issues(
    review: str,
    expected_labels: tuple[str, ...] = SCORECARD_LABELS,
) -> list[str]:
    _ = expected_labels
    text = review.strip()
    if not text:
        return ["empty output"]
    lowered = text.lower()
    if any(marker in lowered for marker in ("<think", "</think", "reasoning trace", "chain-of-thought")):
        return ["contains thinking trace language"]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    issues: list[str] = []
    if not MIN_FIX_COUNT + 2 <= len(lines) <= MAX_FIX_COUNT + 2:
        issues.append(f"expected 3-5 lines, found {len(lines)}")
    if any(not line.startswith("- ") for line in lines):
        issues.append("all lines must be hyphen bullets")

    labels = [_bullet_label(line) for line in lines if line.startswith("- ")]
    if labels:
        fix_count = len(labels) - 2
        expected_label_names = ["strength", *[f"fix {index}" for index in range(1, fix_count + 1)], "next run"]
        if labels != expected_label_names or not MIN_FIX_COUNT <= fix_count <= MAX_FIX_COUNT:
            issues.append("labels must be exactly: Strength, Fix 1[-Fix 3], Next run")
    if lines and not lines[-1].startswith("- Next run:"):
        issues.append("final line must start with Next run")
    return issues


def quote_faithfulness_issues(review: str, transcript: str) -> list[str]:
    transcript_normalized = _normalize_quote_text(transcript)
    if not transcript_normalized:
        return []

    issues: list[str] = []
    for quote in DOUBLE_QUOTE_PATTERN.findall(review):
        quote_normalized = _normalize_quote_text(quote)
        if not _should_check_quote(quote_normalized):
            continue
        if not _quote_matches_transcript(quote_normalized, transcript_normalized):
            issues.append(f'quoted span not found in transcript: "{_truncate_quote(quote)}"')
    return issues


def clean_unfaithful_review_quotes(review: str, transcript: str) -> str:
    transcript_normalized = _normalize_quote_text(transcript)
    if not transcript_normalized:
        return review

    def replace(match: re.Match[str]) -> str:
        quote = match.group(1)
        quote_normalized = _normalize_quote_text(quote)
        if not _should_check_quote(quote_normalized):
            return match.group(0)
        if _quote_matches_transcript(quote_normalized, transcript_normalized):
            return match.group(0)
        return quote

    return DOUBLE_QUOTE_PATTERN.sub(replace, review)


def _quote_matches_transcript(quote_normalized: str, transcript_normalized: str) -> bool:
    return quote_normalized in transcript_normalized


def _should_check_quote(quote_normalized: str) -> bool:
    if not quote_normalized:
        return False
    if len(quote_normalized) < 8:
        return False
    if len(quote_normalized.split()) == 1 and len(quote_normalized) < 12:
        return False
    return True


def _normalize_quote_text(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"[\s,.;:!?()\[\]{}\-]+", " ", text)
    return " ".join(text.split())


def _truncate_quote(quote: str, max_length: int = 90) -> str:
    quote = " ".join(quote.split())
    if len(quote) <= max_length:
        return quote
    return f"{quote[: max_length - 3]}..."


def _bullet_label(line: str) -> str:
    body = line[2:].strip()
    if ":" not in body:
        return ""
    return body.split(":", 1)[0].strip().lower()


def _parse_scorecard_bullet(line: str) -> tuple[str, str]:
    label = _bullet_label(line)
    content = _bullet_content(line)

    nested_label = ""
    if ":" in content:
        nested_label = content.split(":", 1)[0].strip().lower()
    if label in GENERIC_FIX_LABELS and (nested_label in ALLOWED_SCORECARD_LABELS or nested_label in GENERIC_FIX_LABELS):
        nested_content = content.split(":", 1)[1].strip()
        return nested_label, nested_content

    if label in ALLOWED_SCORECARD_LABELS:
        return label, content

    return label, content


def _bullet_content(line: str) -> str:
    body = line[2:].strip()
    if ":" not in body:
        return body
    return body.split(":", 1)[1].strip()


def _build_messages(transcript: str, stats: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _render_prompt("review_system_prompt.jinja2"),
        },
        {
            "role": "user",
            "content": _render_prompt(
                "review_user_prompt.jinja2",
                stats_block=format_review_stats(stats),
                transcript=transcript,
            ),
        },
    ]


def _log_review_prompt(tokenizer: object, messages: list[dict[str, str]]) -> None:
    if not _should_log_review_prompt():
        return

    LOGGER.warning(
        "%s is enabled; logging populated review prompt including transcript and stats.\n%s",
        PROMPT_LOG_ENV,
        _render_chat_prompt_for_logging(tokenizer, messages),
    )


def _render_chat_prompt_for_logging(tokenizer: object, messages: list[dict[str, str]]) -> str:
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - prompt logging should never break review generation.
            LOGGER.warning("Failed to render chat template for prompt logging: %s", exc)
            return _format_messages_for_logging(messages)
    except Exception as exc:  # noqa: BLE001 - prompt logging should never break review generation.
        LOGGER.warning("Failed to render chat template for prompt logging: %s", exc)
        return _format_messages_for_logging(messages)


def _format_messages_for_logging(messages: list[dict[str, str]]) -> str:
    formatted = []
    for message in messages:
        role = str(message.get("role") or "unknown").upper()
        content = str(message.get("content") or "")
        formatted.append(f"{role}:\n{content}")
    return "\n\n".join(formatted)


def _apply_chat_template(tokenizer: object, messages: list[dict[str, str]]) -> object:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )


def _review_generate_kwargs(tokenizer: object) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_REVIEW_TOKENS,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
    return generate_kwargs


def review_speech(transcript: str, stats: Mapping[str, Any] | None = None) -> str:
    text = transcript.strip()
    if not text:
        raise ValueError("The transcript is empty, so there is nothing to review.")

    torch, tokenizer, model = _load_review_stack()
    scorecard_labels = expected_scorecard_labels(text, stats)
    messages = _build_messages(text, stats)
    _log_review_prompt(tokenizer, messages)
    inputs = _apply_chat_template(tokenizer, messages).to(_model_device(model))

    generate_kwargs = _review_generate_kwargs(tokenizer)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    review = clean_review_output(
        tokenizer.decode(generated_ids, skip_special_tokens=True),
        expected_labels=scorecard_labels,
    )
    if not review:
        raise RuntimeError("The review model returned an empty response. Try again with a clearer transcript.")
    shape_issues = scorecard_shape_issues(review, expected_labels=scorecard_labels)
    if shape_issues:
        LOGGER.warning("Review output did not match scorecard shape: %s", "; ".join(shape_issues))
    quote_issues = quote_faithfulness_issues(review, text)
    if quote_issues:
        LOGGER.warning("Review output quoted text not found in transcript: %s", "; ".join(quote_issues))

    return review
