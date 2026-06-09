---
title: Best Man Speech Coach
sdk: gradio
app_file: app.py
python_version: 3.10.4
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

Export Space-compatible dependencies:

```bash
./export_space_requirements.sh
```

## Deployment

The Gradio app is deployed to this Hugging Face Space:

[build-small-hackathon/best-man-speech-practice](https://huggingface.co/spaces/build-small-hackathon/best-man-speech-practice)

GitHub Actions syncs `main` to the Space. The workflow requires a GitHub Actions secret named `HF_TOKEN` with write access to the Space.

The export script pins `torch==2.11.0` in `requirements.txt` because Hugging Face ZeroGPU rejects marker-based multi-version torch entries even when the Linux pin is otherwise compatible.
