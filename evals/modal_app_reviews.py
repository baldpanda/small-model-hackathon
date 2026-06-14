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
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.12.0,<2.0",
        "huggingface-hub>=1.18.0,<2.0",
        "jinja2>=3.1.6,<4.0",
        "safetensors>=0.7.0,<1.0",
        "sentencepiece>=0.2.1,<0.3",
        "torch>=2.10.0,<2.11.0",
        "transformers>=5.6.0,<5.7.0",
    )
    .add_local_file(REPO_ROOT / "review.py", f"{REMOTE_APP_DIR}/review.py", copy=True)
    .add_local_dir(REPO_ROOT / "prompts", f"{REMOTE_APP_DIR}/prompts", copy=True)
)


secrets = []
if os.environ.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HUGGINGFACEHUB_API_TOKEN": os.environ["HUGGINGFACEHUB_API_TOKEN"]}))


app = modal.App("speech-feedback-app-baseline-evals")


@app.function(image=image, gpu=GPU_TYPE, secrets=secrets, timeout=1800)
def generate_reviews_remote(records: list[dict[str, Any]], include_input: bool = False) -> list[dict[str, Any]]:
    sys.path.insert(0, REMOTE_APP_DIR)
    from review import (
        PROMPT_VERSION,
        expected_scorecard_labels,
        quote_faithfulness_issues,
        review_speech,
        scorecard_shape_issues,
    )

    results = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        transcript_id = str(record.get("id") or f"record-{index}")
        transcript = str(record.get("transcript") or "").strip()
        print(f"Generating app review for {transcript_id} ({index}/{total})", flush=True)

        result: dict[str, Any] = {
            "id": transcript_id,
            "review_model": "app.review.review_speech",
            "prompt_version": PROMPT_VERSION,
            "runtime": "modal",
            "gpu": GPU_TYPE,
            "stats": record["stats"],
        }
        try:
            scorecard_labels = expected_scorecard_labels(transcript, record["stats"])
            result["review"] = review_speech(transcript, stats=record["stats"])
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
    output_path: str = "evals/private/modal_app_baseline_reviews.jsonl",
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

    results = generate_reviews_remote.remote(records, include_input=include_input)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    errors = [result for result in results if "error" in result]
    print(f"Wrote {len(results)} rows to {output_file}")
    if errors:
        print(f"{len(errors)} rows contain errors: {[result['id'] for result in errors]}")
