from __future__ import annotations

import os
import logging
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
PROMPT_VERSION = "stats_adaptive_scorecard_v3"
MAX_REVIEW_TOKENS = 260
PROMPTS_DIR = Path(__file__).with_name("prompts")
SCORECARD_LABELS = ("Strength", "Fix 1", "Fix 2", "Next run")
SHORT_SCORECARD_LABELS = ("Strength", "Fix", "Next run")
ALL_SCORECARD_LABELS = tuple(dict.fromkeys((*SCORECARD_LABELS, *SHORT_SCORECARD_LABELS)))
ALLOWED_SCORECARD_LABELS = {label.lower() for label in ALL_SCORECARD_LABELS}
GENERIC_FIX_LABELS = {"fix", "polish"}
SHORT_SCORECARD_MAX_WORDS = 50
SHORT_SCORECARD_MAX_DURATION_SECONDS = 20

LOGGER = logging.getLogger(__name__)
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


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
    if _should_use_short_scorecard(transcript, stats):
        return SHORT_SCORECARD_LABELS
    return SCORECARD_LABELS


def _should_use_short_scorecard(transcript: str, stats: Mapping[str, Any] | None = None) -> bool:
    word_count = _stats_int(stats, "word_count")
    duration_seconds = _stats_float(stats, "duration_seconds")
    if word_count is not None and word_count < SHORT_SCORECARD_MAX_WORDS:
        return True
    if duration_seconds is not None:
        return duration_seconds < SHORT_SCORECARD_MAX_DURATION_SECONDS
    if word_count is not None:
        return False

    return len(transcript.split()) < SHORT_SCORECARD_MAX_WORDS


def _stats_int(stats: Mapping[str, Any] | None, key: str) -> int | None:
    if not stats or stats.get(key) is None:
        return None
    try:
        return int(stats[key])
    except (TypeError, ValueError):
        return None


def _stats_float(stats: Mapping[str, Any] | None, key: str) -> float | None:
    if not stats or stats.get(key) is None:
        return None
    try:
        return float(stats[key])
    except (TypeError, ValueError):
        return None


def clean_review_output(
    review: str,
    expected_labels: tuple[str, ...] = SCORECARD_LABELS,
) -> str:
    review = _strip_thinking_trace(review)
    lines = [line.strip() for line in review.strip().splitlines() if line.strip()]
    bullet_lines = [_normalize_scorecard_line(line) for line in lines]
    bullet_lines = [line for line in bullet_lines if line.startswith("- ")]
    if not bullet_lines:
        return review.strip()

    selected: dict[str, str] = {}

    for line in bullet_lines:
        label, content = _parse_scorecard_bullet(line)
        if label not in ALLOWED_SCORECARD_LABELS and label not in GENERIC_FIX_LABELS:
            continue
        if label == "strength":
            if selected:
                continue
            selected["Strength"] = content
            continue
        if "Strength" not in selected:
            continue
        if "Fix" in expected_labels and label in {"fix", "fix 1", "fix 2", "polish"}:
            selected.setdefault("Fix", content)
            continue
        if label in {"fix 1", "fix 2"}:
            canonical_label = _canonical_label(label)
            selected.setdefault(canonical_label, content)
            continue
        if label in GENERIC_FIX_LABELS:
            fix_label = _next_fix_label(selected)
            if fix_label:
                selected[fix_label] = content
            continue
        if label == "next run":
            selected["Next run"] = content
            break

    if all(label in selected for label in expected_labels):
        return "\n".join(f"- {label}: {selected[label]}" for label in expected_labels)
    return review.strip()


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
    text = review.strip()
    if not text:
        return ["empty output"]
    lowered = text.lower()
    if any(marker in lowered for marker in ("<think", "</think", "reasoning trace", "chain-of-thought")):
        return ["contains thinking trace language"]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    issues: list[str] = []
    if len(lines) != len(expected_labels):
        issues.append(f"expected {len(expected_labels)} lines, found {len(lines)}")
    if any(not line.startswith("- ") for line in lines):
        issues.append("all lines must be hyphen bullets")

    labels = [_bullet_label(line) for line in lines if line.startswith("- ")]
    expected_label_names = [label.lower() for label in expected_labels]
    if labels != expected_label_names:
        issues.append(f"labels must be exactly: {', '.join(expected_labels)}")
    if lines and not lines[-1].startswith("- Next run:"):
        issues.append("final line must start with Next run")
    return issues


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


def _canonical_label(label: str) -> str:
    for canonical in SCORECARD_LABELS:
        if label == canonical.lower():
            return canonical
    raise ValueError(f"Unsupported scorecard label: {label}")


def _next_fix_label(selected: Mapping[str, str]) -> str | None:
    if "Fix 1" not in selected:
        return "Fix 1"
    if "Fix 2" not in selected:
        return "Fix 2"
    return None


def _build_messages(transcript: str, stats: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    scorecard_labels = expected_scorecard_labels(transcript, stats)
    short_scorecard = scorecard_labels == SHORT_SCORECARD_LABELS
    return [
        {
            "role": "system",
            "content": _render_prompt(
                "review_system_prompt.jinja2",
                short_scorecard=short_scorecard,
            ),
        },
        {
            "role": "user",
            "content": _render_prompt(
                "review_user_prompt.jinja2",
                stats_block=format_review_stats(stats),
                transcript=transcript,
                short_scorecard=short_scorecard,
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


def review_speech(transcript: str, stats: Mapping[str, Any] | None = None) -> str:
    text = transcript.strip()
    if not text:
        raise ValueError("The transcript is empty, so there is nothing to review.")

    torch, tokenizer, model = _load_review_stack()
    scorecard_labels = expected_scorecard_labels(text, stats)
    messages = _build_messages(text, stats)
    _log_review_prompt(tokenizer, messages)
    inputs = _apply_chat_template(tokenizer, messages).to(_model_device(model))

    generate_kwargs = {
        "max_new_tokens": MAX_REVIEW_TOKENS,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

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

    return review
