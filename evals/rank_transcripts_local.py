from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "openbmb/MiniCPM5-1B"
REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("rank_transcripts")


SYSTEM_PROMPT = """You rank candidate validation transcripts for a speech-feedback evaluation set.
Use only the transcript and stats provided.
Do not coach the speaker.
Do not invent names, relationships, venues, audience reactions, or delivery details.
Return JSON only with these keys:
validation_value: integer 1-5, where 5 is most useful for held-out evaluation
speech_type: short string
difficulty: one of "low", "medium", "high"
coverage_tags: array of short strings
keep_for_eval: boolean
reasons: array of at most 3 short strings
risks: array of at most 3 short strings"""


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

    torch, tokenizer, model = load_model(args.model_id, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for index, record in enumerate(records, start=1):
            transcript_id = str(record.get("id") or f"record-{index}")
            transcript = str(record.get("transcript") or "").strip()
            if not transcript:
                raise ValueError(f"{transcript_id} has no transcript")

            LOGGER.info("Ranking %s (%s/%s)", transcript_id, index, len(records))
            raw_response = generate_rank(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                transcript_id=transcript_id,
                transcript=transcript,
                stats=record["stats"],
                max_new_tokens=args.max_new_tokens,
                sample=args.sample,
            )
            parsed, parse_error = parse_json_response(raw_response)
            result = {
                "id": transcript_id,
                "stats": record["stats"],
                "rank": parsed,
                "parse_error": parse_error,
                "raw_response": raw_response,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidate speech-validation transcripts with MiniCPM5.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or JSONL with transcript records.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL for ranking results.")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Runtime device. Use cpu if local GPU/MPS setup is fragile.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help="Use MiniCPM no-think sampling instead of greedy output.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def load_model(model_id: str, requested_device: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(torch, requested_device)
    LOGGER.info("Loading %s on %s", model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_kwargs())
    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        **hf_kwargs(),
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return torch, tokenizer, model


def choose_device(torch: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_kwargs() -> dict[str, str]:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(name)
        if value:
            return {"token": value}
    return {}


def generate_rank(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    transcript_id: str,
    transcript: str,
    stats: Any,
    max_new_tokens: int,
    sample: bool,
) -> str:
    user_prompt = build_user_prompt(transcript_id, transcript, stats)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = apply_chat_template(tokenizer, messages).to(model.device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
    }
    if sample:
        generate_kwargs.update({"temperature": 0.7, "top_p": 0.95})
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
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


def build_user_prompt(transcript_id: str, transcript: str, stats: Any) -> str:
    return "\n".join(
        [
            f"Transcript ID: {transcript_id}",
            "",
            "Stats:",
            format_stats(stats),
            "",
            "Transcript:",
            transcript,
        ]
    )


def format_stats(stats: Any) -> str:
    if not isinstance(stats, dict) or not stats:
        return "No stats supplied."
    return json.dumps(stats, ensure_ascii=False, sort_keys=True)


def parse_json_response(response: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed, None

    extracted = extract_json_object(response)
    if extracted is None:
        return None, "response did not contain a JSON object"
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "response JSON was not an object"
    return parsed, None


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


if __name__ == "__main__":
    main()
