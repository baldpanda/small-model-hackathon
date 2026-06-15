---
title: Best Man Speech Coach
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
python_version: "3.12"
tags:
  - build-small-hackathon
  - backyard-ai
  - track:backyard
  - off-the-grid
  - achievement:offgrid
  - well-tuned
  - achievement:welltuned
  - off-brand
  - achievement:offbrand
  - sharing-is-caring
  - achievement:sharing
  - field-notes
  - achievement:fieldnotes
---

# Best Man Speech Coach

Record a roughly one-minute speech rehearsal and get a transcript plus a practical scorecard.
The app reviews structure, pacing, and filler habits so the next run has one or two concrete goals.

**Model budget:** about 3B parameters total across both models. This is a small-model hackathon project, and the small combined budget is part of the build.

- Live Space: [build-small-hackathon/best-man-speech-practice](https://huggingface.co/spaces/build-small-hackathon/best-man-speech-practice)
- Demo video: [YouTube](https://www.youtube.com/watch?v=w7s4LjYW89A)
- Blog post: [Best Man Speech Practice on Hugging Face](https://huggingface.co/blog/pjc43/best-man-speech-practice)

## Models

| Role | Model or artifact | License/status | Notes |
| --- | --- | --- | --- |
| Transcription | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Apache-2.0 | Gated model. Judges or users duplicating the Space must request access on the model page first. |
| Review and feedback | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) | Apache-2.0 | Base review model. The live Space runs a fine-tuned LoRA adapter on top of it. |
| Fine-tuned adapter | [`build-small-hackathon/minicpm5-speech-feedback-lora-v12`](https://huggingface.co/build-small-hackathon/minicpm5-speech-feedback-lora-v12) | Apache-2.0 | LoRA adapter for `openbmb/MiniCPM5-1B`, trained and published for speech-feedback scorecards. |

The review stack uses the MiniCPM5 base model with the published LoRA adapter above. Local runs can fall back to the base model if `REVIEW_ADAPTER_ID` is unset, but the submitted Space is configured to run the fine-tuned version.

## Architecture

1. Record or upload a short rehearsal in Gradio.
2. Transcribe audio on GPU with Cohere Transcribe.
3. Run deterministic timing and filler analysis on CPU.
4. Generate review feedback on GPU with MiniCPM5 plus the LoRA adapter.
5. Render the transcript, pacing notes, filler notes, and scorecard.

## Running The Space

The app runs as a Gradio Space on Hugging Face ZeroGPU. Anonymous users have limited ZeroGPU quota; logging into Hugging Face gives more GPU quota.

The Cohere transcription model is gated. To duplicate or run the Space, request access to [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), then provide a Hugging Face token with access as `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`.

The live review adapter is configured with:

```text
REVIEW_ADAPTER_ID=build-small-hackathon/minicpm5-speech-feedback-lora-v12
```

## Local Development

This project uses `uv` for Python dependency management.

Run the app locally:

```bash
uv run python app.py
```

The app loads both model stacks at startup for ZeroGPU efficiency, so local startup requires access to the gated transcription model through `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`. Set `REVIEW_ADAPTER_ID` to run the LoRA adapter locally.

Validate that the transcription and review models can share one dependency/runtime environment:

```bash
uv run python scripts/prove_phase4_models.py
```

Use `--imports-only`, `--skip-cohere`, or `--skip-minicpm` to narrow the check while debugging. The Cohere transcription check requires `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` because the model is gated.

Export Space-compatible dependencies after dependency changes:

```bash
./export_space_requirements.sh
```

`requirements.txt` is generated for Hugging Face Spaces compatibility. The export script pins `torch==2.10.0` in `requirements.txt` because Hugging Face ZeroGPU rejects marker-based multi-version torch entries even when the Linux pin is otherwise compatible.

## Licensing

Both external models used by the app are Apache-2.0 licensed:

- [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B)

The fine-tuned LoRA adapter [`build-small-hackathon/minicpm5-speech-feedback-lora-v12`](https://huggingface.co/build-small-hackathon/minicpm5-speech-feedback-lora-v12) is also released under Apache-2.0, matching the base model.

This repository currently has no license file. Treat the repository code, specs, and assets as unlicensed unless a license is added later.

## Trace Sanitization

A sanitized snapshot of the Codex traces from this build is published as a Hugging Face dataset for the hackathon's "Sharing is Caring" track: [`build-small-hackathon/best-man-speech-codex-traces`](https://huggingface.co/datasets/build-small-hackathon/best-man-speech-codex-traces). The dataset contains the reviewed traces; the workflow below is what produced them.

Codex trace files can contain prompts, command output, local paths, secrets, private code, and personal details. Keep raw traces under `traces/raw/` and write reviewed copies under `traces/sanitized/`; the whole `traces/` directory is ignored by git.

The sanitizer is a best-effort helper, not a privacy guarantee. Always manually review the sanitized trace before uploading it to a public dataset or Space.

Sanitize a copied Codex JSONL trace:

```bash
python3 scripts/sanitize_codex_trace.py \
  traces/raw/rollout-example.jsonl \
  -o traces/sanitized/rollout-example.sanitized.jsonl
```

By default, the sanitizer redacts common token formats and local paths, strips session base instructions, and drops Codex reasoning items. Add repeated `--redact-term` options for names, venue details, or other private phrases found during manual review:

```bash
python3 scripts/sanitize_codex_trace.py \
  traces/raw/rollout-example.jsonl \
  -o traces/sanitized/rollout-example.sanitized.jsonl \
  --redact-term "private name" \
  --redact-term "private venue"
```

Before publishing a sanitized trace, manually inspect it for private speech details and rerun a secret/path check such as:

```bash
TRACE=traces/sanitized/rollout-example.sanitized.jsonl
rg -n -i 'sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9_-]{20,}|github_pat_|gh[pousr]_|/Users/' "$TRACE"
```

## Deployment

GitHub Actions syncs `main` to the Hugging Face Space. The workflow requires a GitHub Actions secret named `HF_TOKEN` with write access to the Space.
