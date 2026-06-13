---
title: Best Man Speech Coach
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
python_version: "3.12"
---

# Small Model Hackathon

A project for the [Hugging Face Build Small Hackathon](https://huggingface.co/build-small-hackathon).

## About

This repository will contain the code, notes, and experiments for the hackathon build.

## Hackathon Details

- Event: [Hugging Face Build Small Hackathon](https://huggingface.co/build-small-hackathon)
- Dates: June 5 to June 15
- Model size limit: total model size up to 32B parameters
- Submission format: a Gradio app hosted as a Hugging Face Space

## Local Development

This project uses `uv` for Python dependency management.

Run the app locally:

```bash
uv run python app.py
```

The app loads both model stacks at startup for ZeroGPU efficiency, so local app startup requires access to the gated transcription model through `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`.

Validate that the transcription and review models can share one dependency/runtime environment:

```bash
uv run python scripts/prove_phase4_models.py
```

Use `--imports-only`, `--skip-cohere`, or `--skip-minicpm` to narrow the check while debugging. The Cohere transcription check requires `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` because the model is gated.

Export Space-compatible dependencies:

```bash
./export_space_requirements.sh
```

## Trace Sanitization

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

Before publishing a sanitized trace, manually inspect it for private wedding details and rerun a secret/path check such as:

```bash
TRACE=traces/sanitized/rollout-example.sanitized.jsonl
rg -n -i 'sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9_-]{20,}|github_pat_|gh[pousr]_|/Users/' "$TRACE"
```

## Deployment

The Gradio app is deployed to this Hugging Face Space:

[build-small-hackathon/best-man-speech-practice](https://huggingface.co/spaces/build-small-hackathon/best-man-speech-practice)

GitHub Actions syncs `main` to the Space. The workflow requires a GitHub Actions secret named `HF_TOKEN` with write access to the Space.

The export script pins `torch==2.11.0` in `requirements.txt` because Hugging Face ZeroGPU rejects marker-based multi-version torch entries even when the Linux pin is otherwise compatible.
