from __future__ import annotations

import os
from functools import lru_cache


MODEL_ID = "openbmb/MiniCPM5-1B"
MAX_REVIEW_TOKENS = 360


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
def _load_review_stack() -> tuple[object, object, object]:
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
        device_map="auto",
        **_model_kwargs(),
    )
    return torch, tokenizer, model


def _build_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a supportive best man speech rehearsal coach. Give practical, specific feedback. "
                "Do not rewrite the whole speech. Preserve the speaker's voice and personal stories."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this rehearsal transcript for a best man speech.\n\n"
                "Return four short sections:\n"
                "Overall impression\n"
                "What is working\n"
                "What to improve\n"
                "Next rehearsal checklist\n\n"
                f"Transcript:\n{transcript}"
            ),
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

    torch, tokenizer, model = _load_review_stack()
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
