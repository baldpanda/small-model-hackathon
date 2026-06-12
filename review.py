from __future__ import annotations

import os
import logging
import time
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


MODEL_ID = "openbmb/MiniCPM5-1B"
MAX_REVIEW_TOKENS = 360
PROMPTS_DIR = Path(__file__).with_name("prompts")

LOGGER = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def _prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def _render_prompt(template_name: str, **context: str) -> str:
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
    runtime_device = _place_model_for_runtime(torch, model)
    model.eval()
    LOGGER.info(
        "Loaded review stack on %s in %.1fs",
        runtime_device,
        time.perf_counter() - load_started_at,
    )
    return torch, tokenizer, model


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


_REVIEW_STACK = _load_review_stack()


def _build_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _render_prompt("review_system_prompt.jinja2"),
        },
        {
            "role": "user",
            "content": _render_prompt("review_user_prompt.jinja2", transcript=transcript),
        },
    ]


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


def review_speech(transcript: str) -> str:
    text = transcript.strip()
    if not text:
        raise ValueError("The transcript is empty, so there is nothing to review.")

    torch, tokenizer, model = _REVIEW_STACK
    inputs = _apply_chat_template(tokenizer, _build_messages(text)).to(model.device)

    generate_kwargs = {
        "max_new_tokens": MAX_REVIEW_TOKENS,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    review = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not review:
        raise RuntimeError("The review model returned an empty response. Try again with a clearer transcript.")

    return review
