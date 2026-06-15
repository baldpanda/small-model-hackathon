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

The variable-fix prompt baseline is tagged as `stats_variable_fix_scorecard_v4`. The app-facing scorecard uses `Strength`, `Fix 1`, optional `Fix 2` and `Fix 3`, and `Next run`. The model should use one fix by default and add more only when each additional fix is clearly useful and distinct. Stats are included in the input but do not get a reserved slot. The app path uses deterministic decoding plus a small repetition penalty, then applies a scorecard cleaner that deduplicates repeated fixes, renumbers the remaining fixes, drops extra bullets if the model continues after the next-step line, and appends a conservative fallback `Next run` when the model gives a strength and fix but omits the final action. It also applies a transcript-grounded faithfulness cleaner for observed high-risk paraphrase errors: actor/ownership perspective flips around first-person events, negation flips around `would do anything`, unsupported marriage-pronoun flips, unfaithful quote marks, and a few narrowly observed typo/ASR cleanup cases. Eval outputs include `scorecard_shape_valid`, `scorecard_shape_issues`, `quote_faithfulness_valid`, and `quote_faithfulness_issues`; there is no retry or repair generation pass. Do not add `no_repeat_ngram_size=3` here: a held-out Modal run showed it corrupts fixed labels into invalid values such as `Fix 4`.

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

Before a real training run, also check that the held-out eval inputs and SFT inputs use the same core stats-block schema:

```bash
python evals/check_stats_schema_consistency.py \
  --eval-input evals/data/eval_transcripts.csv \
  --sft-input evals/private/sft_train_messages.jsonl \
  --sft-input evals/private/sft_val_messages.jsonl
```

The SFT validator prints the assistant fix-count distribution. Do not start the full run until the gold data includes enough single-fix rows to teach restraint on short or already-strong speeches. You can enforce a minimum while iterating:

```bash
python evals/validate_sft_dataset.py \
  --input evals/private/sft_train_messages.jsonl \
  --min-single-fix-rows 20
```

Before the next real training run, fix assistant quotes and enable the quote-faithfulness gate:

```bash
python evals/validate_sft_dataset.py \
  --input evals/private/sft_train_messages.jsonl \
  --check-quote-faithfulness \
  --check-distinct-fixes
python evals/validate_sft_dataset.py \
  --input evals/private/sft_val_messages.jsonl \
  --check-quote-faithfulness \
  --check-distinct-fixes
```

This fails when a gold response puts a paraphrase in double quotes. Use double quotes only for exact transcript spans; paraphrase without quotation marks.
It also fails when two gold fix bullets are duplicate or near-duplicate, because the model has repeatedly inherited redundant-fix patterns from small SFT data.

The split report includes scenario-family counts when rows provide `scenario_family`, `family_id`, or `family`. Keep variants of the same hard case in the same family so the train/validation split does not leak near-duplicate examples across both sides.

The base gold split is 104 train rows, 26 validation rows, and an 8-row smoke sample drawn from train. Targeted augment rows are added to train only by default and should be evaluated with a separate held-out canary JSONL.

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
  --max-steps 2 \
  --gradient-accumulation-steps 2
```

If the smoke adapter is created successfully, run the full training job. With 104 training rows, this uses roughly 13 optimizer updates per epoch and about 52 total updates:

```bash
modal run evals/modal_train_minicpm5_lora.py \
  --run-name full \
  --data-rel-path data/sft_train_messages.jsonl \
  --val-rel-path data/sft_val_messages.jsonl \
  --epochs 4 \
  --max-steps 0 \
  --gradient-accumulation-steps 2
```

Each Modal training run writes these inspection artifacts under `runs/<run-name>/` in the Modal Volume:

- `assistant_mask_check.json`, which confirms the loss mask covers assistant feedback tokens and not the transcript/stats prompt
- `trainer_state.json`
- `train_metrics.json`
- `log_history.json`
- `adapter_final/`

Pull the final adapter locally for private evaluation:

```bash
modal volume get speech-feedback-minicpm5-ft runs/full/adapter_final adapters/minicpm5-speech-feedback-lora --force
```

Pull training diagnostics when comparing runs:

```bash
modal volume get speech-feedback-minicpm5-ft runs/full/trainer_state.json evals/private/trainer_state_full.json --force
modal volume get speech-feedback-minicpm5-ft runs/full/log_history.json evals/private/log_history_full.json --force
modal volume get speech-feedback-minicpm5-ft runs/full/assistant_mask_check.json evals/private/assistant_mask_check_full.json --force
```

## Evaluate The Adapter

Run the adapter against the same held-out eval set used for the optimized prompt baseline:

```bash
modal run evals/modal_adapter_reviews.py \
  --input-path evals/data/eval_transcripts.csv \
  --output-path evals/private/modal_adapter_reviews.jsonl \
  --adapter-rel-path runs/full/adapter_final
```

Treat validation loss as a training sanity check, not the product metric. Select between candidate adapters by scoring the held-out reviews with `evals/speech-feedback-coaching-rubric.md`, including hard-gate rate, per-dimension deltas, invented-detail failures, stats-use failures, scorecard-shape failures, and repeated/non-distinct fixes.

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
