# Phase 2 Foundation Spec

## Goal

Prove the delivery path before adding model complexity.

Phase 2 should create the smallest possible Gradio app, run it locally, host it on Hugging Face Spaces, and establish a basic CI/CD workflow so future commits can update the hosted app predictably.

This phase is deliberately not about speech transcription or model feedback yet. It is about making sure the project can move from local code to a live, shareable app.

## Why This Comes First

The hackathon submission must be a Gradio app hosted as a Hugging Face Space. Before building the speech practice logic, we should de-risk the app shell, dependency setup, deployment path, and automation.

This supports the manifesto principle: get end-to-end early.

## User Story

As a builder, I want to run a tiny Gradio app locally, see the same app hosted on Hugging Face Spaces, and trust that a commit can update the hosted version, so that later model work lands inside a working delivery pipeline.

## Scope

Phase 2 includes:

- A minimal Gradio app.
- Local run instructions.
- Hugging Face Space-ready project files.
- A hosted Space running the app.
- A basic CI/CD path from GitHub to Hugging Face.
- Documentation of the deployment setup.

Phase 2 does not include:

- Speech-to-text integration.
- LLM review.
- Filler word analysis.
- Timing analysis.
- Full UI polish.
- Production-grade release management.

## Minimal App Behavior

The first app should be intentionally simple.

Required behavior:

- The user can enter a short piece of text.
- The app returns a friendly response confirming the app is running.
- The response should make clear that this is the best man speech practice project.

Example:

- Input: `I'm testing the app.`
- Output: `Speech Coach is alive. You said: "I'm testing the app."`

This is a smoke test, not the final product experience.

## Local Development

The project should support a simple local run flow:

```bash
uv run python app.py
```

Local setup should be documented in the README.

Expected local files:

- `app.py`: minimal Gradio app.
- `pyproject.toml`: Python project metadata and dependencies managed with `uv`.
- `uv.lock`: locked dependency versions.
- `requirements.txt`: generated from `uv` for Hugging Face Spaces compatibility.
- `README.md`: run and deployment notes.

## Hugging Face Space

The app should be deployable as a Hugging Face Space.

Expected Space configuration:

- SDK: Gradio
- Runtime: Python
- Entry point: `app.py`
- Dependencies: managed with `uv`, with `requirements.txt` exported for the Space build.

The hosted Space should be usable by someone with the link.

## CI/CD

The first automation goal is simple: GitHub should be the source of truth, and changes merged to `main` should update the Hugging Face Space without manual file copying.

Chosen approach:

- Use GitHub Actions to sync the repository to the Hugging Face Space.
- Use the official `huggingface/hub-sync` action unless a custom workflow becomes necessary.
- Store a fine-grained Hugging Face write token as a GitHub secret named `HF_TOKEN`.
- Run deployment on pushes to `main`, which means normal PR merges trigger deployment.
- Export `requirements.txt` from `uv.lock` before syncing so the Space installs the same dependency versions tested locally.

## Acceptance Criteria

- A developer can run the app locally with a documented command.
- Dependencies are managed with `uv`.
- The local app opens a Gradio interface.
- The app has a tiny working interaction.
- The repository contains the files required for a Gradio Hugging Face Space.
- A Hugging Face Space exists and runs the app.
- GitHub Actions syncs the repository to the Hugging Face Space on pushes to `main`.
- `HF_TOKEN` is documented as the required GitHub secret.
- `requirements.txt` is generated from the `uv` lockfile for Space compatibility.

## Open Questions

- What should the Hugging Face Space be called?
- Should the Space be public from the start?
- Should deployment require any manual approval, or is automatic deployment on `main` enough for the hackathon?

## Implementation Sketch

1. Add a minimal `app.py`.
2. Add `pyproject.toml` with Gradio as a dependency.
3. Generate `uv.lock`.
4. Generate `requirements.txt` from `uv`.
5. Update `README.md` with local run instructions.
6. Create the Hugging Face Space.
7. Add a GitHub Actions workflow that exports dependencies and syncs to the Space.
8. Add `HF_TOKEN` as a GitHub secret.
9. Verify the hosted app manually.

## Definition of Done

Phase 2 is done when the app can say hello locally and on Hugging Face Spaces, and the repository explains how the deployed version gets updated.

Once this is true, Phase 3 can focus on replacing the hello-world interaction with the real speech practice workflow from the Phase 1 spec.
