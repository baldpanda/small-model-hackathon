# Local MiniCPM Eval Environment

This folder has a minimal local environment for validation-data curation and baseline review generation with `openbmb/MiniCPM5-1B`.

It is intentionally separate from the deployed app environment. Do not install the root `pyproject.toml` dependencies for this workflow; that brings in the Gradio, Spaces, audio, and transcription stack.

## Native Python

On Apple Silicon, make sure the terminal and Python are native arm64, not running under Rosetta:

```bash
python3 -c "import platform; print(platform.machine())"
```

Expected output on Apple Silicon is `arm64`. If it prints `x86_64`, switch to a native arm64 Python before installing Torch.

The repository root currently asks pyenv for Python 3.12. If that is not installed locally, use a native Homebrew Python directly:

```bash
/opt/homebrew/bin/python3 -m venv evals/.venv-minicpm
```

If you have another native Python 3.10+ available, use that instead. Do not use the local pyenv `3.10.4` on this machine unless it reports `arm64`; it currently reports `x86_64`.

## Install

```bash
source evals/.venv-minicpm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evals/requirements-minicpm.txt
```

For Modal GPU evals, also install:

```bash
python -m pip install -r evals/requirements-modal.txt
```

## Input Format

Use CSV or JSONL. For JSONL, each line should be one candidate transcript:

```json
{"id":"speech-001","transcript":"Good evening everyone...","stats":{"word_count":420,"wpm":168,"duration_seconds":150,"filler_counts":{"um":3,"like":2}}}
```

Only `transcript` is required. `id` and duration fields are strongly recommended.

The eval loader computes word count and filler stats from the transcript using the same repo logic as the app. If a duration is present, it also derives WPM and filler-per-minute. Supplied CSV stat columns are treated as source data for inspection, not as the eval source of truth.

Keep private validation data under `evals/private/`, which is ignored by git.

## Generate App Baseline Reviews

For base-model scoring, use the same review path as the Hugging Face app:

```bash
python evals/generate_app_reviews_local.py \
  --input evals/data/eval_transcripts.csv \
  --output evals/private/app_baseline_reviews.jsonl
```

This imports `review.review_speech()` from the app code, so it uses the same model ID, prompt templates, generation settings, deterministic stats block, and `enable_thinking=False` behavior as the deployed app review path.

The adaptive-shape prompt baseline is tagged as `stats_adaptive_scorecard_v3`. Substantive clips use the `Strength`/`Fix 1`/`Fix 2`/`Next run` scorecard. Clips under 50 words or under 20 seconds use the shorter `Strength`/`Fix`/`Next run` scorecard so the model does not force a second fix from thin material. The app path applies a narrow scorecard cleaner after generation and drops extra bullets if the model continues after the next-step line. Eval outputs include `scorecard_shape_valid` and `scorecard_shape_issues`; there is no retry or repair generation pass.

## Run App Baseline Reviews On Modal

Local CPU generation can be slow or stall. To run the same app review path on a Modal GPU:

```bash
modal setup
modal run evals/modal_app_reviews.py \
  --input-path evals/data/eval_transcripts.csv \
  --output-path evals/private/modal_app_optimized_reviews.jsonl
```

The Modal job defaults to `L4`. To use a larger GPU:

```bash
MODAL_GPU=L40S modal run evals/modal_app_reviews.py \
  --input-path evals/data/eval_transcripts.csv \
  --output-path evals/private/modal_app_optimized_reviews.jsonl
```

The remote function packages only `review.py` and `prompts/`, then calls `review.review_speech(transcript, stats=...)` for each transcript. The CSV is read locally, normalized into transcript-plus-stats records, and sent to the remote function; no private data is committed.

## Prepare Fine-Tuning Data

The labelled training source is `evals/data/master_transcripts_gold.csv`. Join it to the private computed stats CSV and write private messages-format JSONL:

```bash
python evals/prepare_sft_dataset.py \
  --gold-csv evals/data/master_transcripts_gold.csv \
  --stats-csv evals/private/master_transcript_stats.csv \
  --eval-csv evals/data/eval_transcripts.csv \
  --output-dir evals/private
```

This writes:

- `evals/private/sft_smoke_messages.jsonl`
- `evals/private/sft_train_messages.jsonl`
- `evals/private/sft_val_messages.jsonl`
- `evals/private/sft_split_report.json`

Validate each split before uploading to Modal:

```bash
python evals/validate_sft_dataset.py --input evals/private/sft_smoke_messages.jsonl
python evals/validate_sft_dataset.py --input evals/private/sft_train_messages.jsonl
python evals/validate_sft_dataset.py --input evals/private/sft_val_messages.jsonl
```

The current default split is 104 train rows, 26 validation rows, and an 8-row smoke sample drawn from train.

## Fine-Tune On Modal

Create the Modal Volume once:

```bash
modal volume create speech-feedback-minicpm5-ft
```

Upload the prepared JSONL files:

```bash
modal volume put speech-feedback-minicpm5-ft evals/private/sft_smoke_messages.jsonl data/sft_smoke_messages.jsonl --force
modal volume put speech-feedback-minicpm5-ft evals/private/sft_train_messages.jsonl data/sft_train_messages.jsonl --force
modal volume put speech-feedback-minicpm5-ft evals/private/sft_val_messages.jsonl data/sft_val_messages.jsonl --force
```

Run a smoke training job first:

```bash
modal run evals/modal_train_minicpm5_lora.py \
  --run-name smoke \
  --data-rel-path data/sft_smoke_messages.jsonl \
  --epochs 1 \
  --limit 8 \
  --max-steps 2
```

If the smoke adapter is created successfully, run the full training job:

```bash
modal run evals/modal_train_minicpm5_lora.py \
  --run-name full \
  --data-rel-path data/sft_train_messages.jsonl \
  --val-rel-path data/sft_val_messages.jsonl \
  --epochs 2 \
  --max-steps 0
```

Pull the final adapter locally for private evaluation:

```bash
modal volume get speech-feedback-minicpm5-ft runs/full/adapter_final adapters/minicpm5-speech-feedback-lora --force
```

## Evaluate The Adapter

Run the adapter against the same held-out eval set used for the optimized prompt baseline:

```bash
modal run evals/modal_adapter_reviews.py \
  --input-path evals/data/eval_transcripts.csv \
  --output-path evals/private/modal_adapter_reviews.jsonl \
  --adapter-rel-path runs/full/adapter_final
```

For a smoke inference check, use the smoke adapter and a small limit:

```bash
modal run evals/modal_adapter_reviews.py \
  --input-path evals/data/eval_transcripts.csv \
  --output-path evals/private/modal_adapter_smoke_reviews.jsonl \
  --adapter-rel-path runs/smoke/adapter_final \
  --limit 2
```

## Rank Candidate Transcripts

```bash
python evals/rank_transcripts_local.py \
  --input evals/data/eval_transcripts.csv \
  --output evals/private/ranked_transcripts.jsonl \
  --device cpu
```

Use `--device auto` to try CUDA or MPS when available. CPU is slower but usually the least fragile local option.

The ranker is a curation helper for choosing useful held-out transcripts. It is not the app baseline scorer.
