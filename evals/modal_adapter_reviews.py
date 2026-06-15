from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
REMOTE_APP_DIR = "/root/app"
REMOTE_VOLUME_DIR = "/vol"
VOLUME_NAME = "speech-feedback-minicpm5-ft"
MODEL_ID = "openbmb/MiniCPM5-1B"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.12.0,<2.0",
        "huggingface-hub>=1.18.0,<2.0",
        "jinja2>=3.1.6,<4.0",
        "peft>=0.13.0,<1.0",
        "safetensors>=0.7.0,<1.0",
        "sentencepiece>=0.2.1,<0.3",
        "torch>=2.11.0",
        "transformers>=5.6.0,<6.0",
    )
    .add_local_file(REPO_ROOT / "review.py", f"{REMOTE_APP_DIR}/review.py", copy=True)
    .add_local_dir(REPO_ROOT / "prompts", f"{REMOTE_APP_DIR}/prompts", copy=True)
)


secrets = []
if os.environ.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HUGGINGFACEHUB_API_TOKEN": os.environ["HUGGINGFACEHUB_API_TOKEN"]}))


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("speech-feedback-minicpm5-adapter-evals")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=secrets,
    timeout=1800,
    volumes={REMOTE_VOLUME_DIR: volume},
)
def generate_adapter_reviews_remote(
    records: list[dict[str, Any]],
    *,
    adapter_rel_path: str,
    include_input: bool = False,
    max_new_tokens: int = 260,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, REMOTE_APP_DIR)
    from review import (
        PROMPT_VERSION,
        _build_messages,
        _review_generate_kwargs,
        clean_review_output,
        expected_scorecard_labels,
        quote_faithfulness_issues,
        scorecard_shape_issues,
    )

    adapter_path = str(Path(REMOTE_VOLUME_DIR) / adapter_rel_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **hf_kwargs())
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **hf_kwargs(),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()

    results = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        transcript_id = str(record.get("id") or f"record-{index}")
        transcript = str(record.get("transcript") or "").strip()
        print(f"Generating adapter review for {transcript_id} ({index}/{total})", flush=True)
        result: dict[str, Any] = {
            "id": transcript_id,
            "review_model": "peft.openbmb/MiniCPM5-1B",
            "prompt_version": PROMPT_VERSION,
            "adapter_rel_path": adapter_rel_path,
            "runtime": "modal",
            "gpu": GPU_TYPE,
            "stats": record["stats"],
        }
        try:
            scorecard_labels = expected_scorecard_labels(transcript, record["stats"])
            messages = _build_messages(transcript, record["stats"])
            inputs = apply_chat_template(tokenizer, messages).to(model.device)
            generate_kwargs = _review_generate_kwargs(tokenizer)
            generate_kwargs["max_new_tokens"] = max_new_tokens
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generate_kwargs)
            generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            result["review"] = clean_review_output(
                tokenizer.decode(generated_ids, skip_special_tokens=True),
                expected_labels=scorecard_labels,
                transcript=transcript,
            )
            result["scorecard_shape_issues"] = scorecard_shape_issues(
                result["review"],
                expected_labels=scorecard_labels,
            )
            result["scorecard_shape_valid"] = not result["scorecard_shape_issues"]
            result["quote_faithfulness_issues"] = quote_faithfulness_issues(result["review"], transcript)
            result["quote_faithfulness_valid"] = not result["quote_faithfulness_issues"]
        except Exception as exc:  # noqa: BLE001 - eval rows should capture failures and continue.
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)

        if include_input:
            result["transcript"] = transcript
            for key in ("category", "speech_type", "scenario_notes", "has_garble", "gold_feedback"):
                if key in record:
                    result[key] = record[key]
        results.append(result)

    return results


@app.local_entrypoint()
def main(
    input_path: str = "evals/data/eval_transcripts.csv",
    output_path: str = "evals/private/modal_adapter_reviews.jsonl",
    adapter_rel_path: str = "runs/full/adapter_final",
    include_input: bool = True,
    limit: int | None = None,
) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EVALS_DIR))
    from eval_data import read_eval_records

    input_file = Path(input_path)
    output_file = Path(output_path)
    records = read_eval_records(input_file)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise SystemExit(f"No records found in {input_file}")

    results = generate_adapter_reviews_remote.remote(
        records,
        adapter_rel_path=adapter_rel_path,
        include_input=include_input,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    errors = [result for result in results if "error" in result]
    print(f"Wrote {len(results)} rows to {output_file}")
    if errors:
        print(f"{len(errors)} rows contain errors: {[result['id'] for result in errors]}")


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


def hf_kwargs() -> dict[str, str]:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return {"token": value}
    return {}
