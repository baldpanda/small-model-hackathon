# Phase 10 MiniCPM5 Finetuning Spec

## Goal

Produce and evaluate a speech-feedback LoRA adapter for `openbmb/MiniCPM5-1B`.

Phase 10 should turn the current prompt-only MiniCPM review loop into an evaluation-led fine-tuning experiment. The phase should define a rubric for speech feedback, create private hand-written evaluation transcripts, run the existing review path as the baseline, optimize the prompt, generate roughly 130 private synthetic training examples locally, fine-tune a LoRA adapter on Modal, and compare the adapter against the frozen baseline before any app integration.

This phase is about proving whether fine-tuning improves feedback quality. It is not about swapping the app to a fine-tuned model by default.

## Why This Comes Next

The app now has a working two-minute rehearsal loop with transcription, timing feedback, filler analysis, and MiniCPM-generated speech feedback.

The next product risk is quality rather than feature coverage. The current prompt can produce useful feedback, but a small task-specific adapter may make the review more consistent, more specific, and less likely to drift into generic wedding-speech advice.

This phase de-risks:

- a clear rubric for speech-feedback quality
- a concrete scorecard output contract for evaluation and training
- repeatable evaluation against the current `review_speech` path
- prompt optimization before training
- private dataset creation for speech coaching
- OpenBMB MiniCPM5 LoRA fine-tuning
- Modal as the remote training runtime
- evaluation before deployment

## User Story

As someone rehearsing a speech, I want the app's feedback to be consistently specific, practical, and true to what I actually said, so I can improve without the model flattening my voice or inventing context.

## Scope

Phase 10 includes:

- defining a rubric for speech-feedback quality
- using `evals/speech-feedback-coaching-rubric.md` as the rubric source of truth
- using transcript plus deterministic stats as the fine-tuning and evaluation input
- hand-writing private held-out evaluation transcripts
- running the existing `review_speech(transcript: str) -> str` path as the baseline
- evaluating with `openbmb/MiniCPM5-1B` and thinking mode off
- optimizing the existing prompt templates before fine-tuning
- generating about 130 private synthetic chat-format SFT examples locally
- validating generated examples before training
- fine-tuning a LoRA adapter on Modal
- evaluating the adapter against the current prompt baseline and optimized prompt baseline

Phase 10 does not include:

- using Codex Cloud for synthetic data generation
- committing raw transcripts or generated training data
- uploading a public dataset
- wiring the LoRA adapter into `app.py`
- changing the live Hugging Face Space behavior
- full-model fine-tuning
- DPO, RLHF, GRPO, or preference training
- changing the transcription model
- rewriting the deterministic timing or filler analysis

## Model Behavior

Chosen model:

- `openbmb/MiniCPM5-1B`
- Model card: https://huggingface.co/openbmb/MiniCPM5-1B

The model should run with thinking mode off for this product surface.

Required inference behavior:

- use the OpenBMB MiniCPM5 chat template where available
- pass `enable_thinking=False` for baseline inference, optimized-prompt evaluation, adapter evaluation, and any future app-facing inference from this phase
- keep output concise and user-facing
- do not expose reasoning traces
- train and evaluate against the compact scorecard output contract in `evals/speech-feedback-coaching-rubric.md`

Thinking mode should stay off because this workflow needs direct coaching output, not visible reasoning. The fine-tuned adapter should learn the final feedback style, not chain-of-thought or hidden deliberation.

## Feedback Rubric

The rubric source of truth is `evals/speech-feedback-coaching-rubric.md`.

The rubric uses 0/1/2 scoring across dimensions for context tailoring, content specificity, structural insight, voice and humor preservation, emotional or occasion turn, delivery and stats translation, proportionality, earned opening strength, actionable next step, and no invented details.

The rubric also defines hard gates for material hallucination, reasoning traces, full-speech rewriting, generic wedding advice on non-wedding transcripts, too many fixes, missing next step, and ignoring relevant stats.

## Evaluation Workflow

Create a private held-out transcript set before generating training examples.

The held-out set should include:

- best man speeches
- wedding toasts that are not best man speeches
- general celebration toasts
- Toastmasters or meeting-role introductions
- very short clips with limited material
- rambling clips with weak structure
- strong but improvable speeches

Evaluation steps:

1. Run held-out transcripts through the current `review_speech` path as the current app baseline.
2. Score each output with the rubric and record failure modes.
3. Optimize the existing prompt templates.
4. Freeze the best prompt baseline.
5. Run the same held-out transcripts through the optimized prompt baseline.
6. Train the LoRA adapter.
7. Run the same held-out transcript-plus-stats examples through the adapter with `enable_thinking=False`.
8. Compare current baseline, optimized prompt baseline, and LoRA adapter outputs.

The held-out evaluation transcripts must not be included in training data.

## Synthetic Training Data

Synthetic data generation should happen locally and privately, not in Codex Cloud.

Target dataset:

- about 130 accepted chat-format SFT examples
- JSONL format
- each row has a stable example ID, category metadata, and `messages`
- each `messages` value contains `system`, `user`, and `assistant` messages
- the `user` message should include transcript plus deterministic stats
- the `system` and `user` messages should match the no-thinking speech-review behavior described in the rubric
- the `assistant` message should be the desired final feedback, not reasoning

Recommended category mix:

- best man and wedding speeches
- general toasts and celebration speeches
- functional speeches and Toastmasters-style role introductions
- short or incomplete transcripts
- rambling or poorly structured transcripts
- strong but improvable transcripts
- edge cases where the model must avoid inventing context

Generated examples should be validated for:

- schema correctness
- valid message roles
- unique IDs
- train/eval separation
- duplicate or near-duplicate transcripts
- generic feedback
- hallucinated context
- full-speech rewrites
- real names, venue details, dates, or private stories
- thinking traces or hidden-reasoning style content

Raw generated JSONL should remain in an ignored private data directory. Only reusable schemas, generation prompts, validation scripts, and sanitized fixtures should be committed if needed.

## Modal Finetuning

Training should run on Modal, not in the Hugging Face Space.

Use OpenBMB's TRL + PEFT LoRA recipe for MiniCPM5-1B:

- `SFTTrainer`
- `LoraConfig`
- assistant-only loss
- the training-only chat-template patch with a generation block
- original MiniCPM5 tokenizer reloaded for inference and evaluation

Training defaults should start conservative:

- LoRA, not full-model fine-tuning
- BF16 where supported
- bounded context length suitable for two-minute rehearsal transcripts
- small number of epochs
- explicit seed
- saved PEFT adapter artifacts

Modal expectations:

- use a Modal GPU function for training
- use Modal Secrets for Hugging Face access if needed
- use Modal Volumes for training data copies, checkpoints, logs, and adapter artifacts
- run a tiny smoke training job before the full run
- persist the final adapter as PEFT files such as `adapter_model.safetensors` and `adapter_config.json`

Merging the adapter into a full model is optional and not required for this phase.

## Privacy and Storage

Private speech data should not be committed.

Keep these out of git:

- raw hand-written transcripts
- generated training JSONL
- held-out evaluation inputs
- model outputs from private transcripts
- Modal training logs when they contain transcript text
- adapter artifacts

Use sanitized examples only when a committed fixture is necessary for scripts or documentation. Sanitized fixtures must not contain real names, wedding dates, venues, private stories, or other personal details.

## Acceptance Criteria

- The phase spec exists and clearly names `openbmb/MiniCPM5-1B` as the fine-tuning target.
- The spec requires thinking mode off with `enable_thinking=False`.
- The rubric is defined before training in `evals/speech-feedback-coaching-rubric.md`.
- A private held-out evaluation set is created before synthetic training examples.
- The current `review_speech` path is measured as the baseline.
- The prompt is optimized and frozen before LoRA training.
- About 130 accepted private training examples are generated locally.
- The dataset passes schema, privacy, duplication, and quality checks.
- A Modal smoke training job produces a valid PEFT adapter.
- The full Modal training run produces a final LoRA adapter.
- The adapter is evaluated against the same held-out transcripts as the baselines.
- The adapter is not wired into the production app during this phase.

## Implementation Sketch

1. Add this phase spec.
2. Add ignored local directories for private phase-10 data and artifacts.
3. Define the rubric and private evaluation transcript template.
4. Add a local schema and validator for SFT JSONL examples.
5. Create and score the private held-out evaluation set.
6. Run baseline and optimized-prompt evaluations.
7. Generate and validate roughly 130 private training examples locally.
8. Add a Modal training entrypoint based on OpenBMB's TRL + PEFT recipe.
9. Run a Modal smoke training job.
10. Run the full Modal LoRA training job.
11. Evaluate the adapter with thinking mode off.
12. Summarize whether the adapter should be considered for a later app-integration phase.

## Open Questions

- What is the smallest held-out evaluation set that is still useful for this hackathon timeline?
- Should the adapter evaluation use only human rubric scores, or also a structured model-assisted judge after manual spot checks?
- Should generated training examples require four bullets exactly, or allow a small number of tight one-paragraph scorecards?
- Which Modal GPU offers the best cost and latency tradeoff for a small MiniCPM5 LoRA run?

## Definition of Done

Phase 10 is done when there is a private, validated MiniCPM5 LoRA adapter trained on Modal and a rubric-based comparison showing whether it improves on the current and optimized prompt baselines.

The live app should remain on the existing prompt-only review path until a later phase explicitly approves adapter integration.
