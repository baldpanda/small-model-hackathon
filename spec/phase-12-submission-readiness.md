# Phase 12 Submission Readiness Spec

## Goal

Prepare the project for hackathon submission by making model attribution, parameter budget, runtime notes, licensing, links, and repository hygiene clear.

This phase is documentation and cleanup only. It should not change the app's runtime behavior, model loading flow, prompt contract, or scorecard output.

## Why This Comes Next

The live app now uses two small models in one rehearsal workflow:

- transcription with `CohereLabs/cohere-transcribe-03-2026`
- review generation with `openbmb/MiniCPM5-1B` plus the published LoRA adapter `build-small-hackathon/minicpm5-speech-feedback-lora-v5`

For submission, judges should be able to understand the model story quickly, duplicate the Space with the right gated-model access, and verify that the repository does not expose private data or secrets.

## Documentation Requirements

The README is the primary documentation surface and also acts as the Hugging Face Space card when synced to the Space. If a separate Space card is introduced later, it must contain the same model attribution and runtime notes.

The README should state prominently:

- the app records about a one-minute speech rehearsal and returns a transcript plus scorecard for structure, pacing, and filler habits
- the total model budget is about 3B parameters across both models
- this is a small-model hackathon project and the parameter budget is part of the project story
- the live Space URL is `https://huggingface.co/spaces/build-small-hackathon/best-man-speech-practice`
- the demo video URL is still to be filled in if the final video link is not available

The README model attribution must include:

- Transcription: `CohereLabs/cohere-transcribe-03-2026`
  - model page: `https://huggingface.co/CohereLabs/cohere-transcribe-03-2026`
  - license: Apache-2.0
  - note that the model is gated
  - note that anyone duplicating the Space, including judges, must request access on the model page first
- Review and feedback: `openbmb/MiniCPM5-1B`
  - model page: `https://huggingface.co/openbmb/MiniCPM5-1B`
  - license: Apache-2.0
  - note that the live Space runs a fine-tuned LoRA adapter on top of this base model
- Fine-tuned adapter:
  - adapter page: `https://huggingface.co/build-small-hackathon/minicpm5-speech-feedback-lora-v5`
  - describe it as the LoRA adapter trained and published for speech-feedback scorecards

The README runtime section should explain:

- the app runs as a Gradio Space on Hugging Face ZeroGPU
- anonymous users have limited ZeroGPU quota
- logging into Hugging Face gives more GPU quota
- the architecture is: transcription on GPU, deterministic filler and timing analysis on CPU, review generation on GPU, then scorecard rendering

The README licensing section should state:

- both external models are Apache-2.0
- this repository currently has no license file and should be treated as unlicensed unless a license is added later

The README front matter should remain valid for the Space:

```yaml
---
title: Best Man Speech Coach
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
python_version: "3.12"
---
```

## Repository Hygiene Requirements

Before submission, verify that no secrets, tokens, or private values are committed.

At minimum, scan tracked files for:

- Hugging Face tokens such as `hf_...`
- `HF_TOKEN=` or `HUGGINGFACEHUB_API_TOKEN=` assignments
- GitHub tokens such as `github_pat_...` and `ghp_...`
- OpenAI-style `sk-...` tokens
- generic long token or secret assignments

Documentation examples and redaction regexes are allowed only when they do not include real token values.

The `.gitignore` should clearly cover private or generated material that must not be committed:

- private training data
- private transcripts
- held-out eval inputs
- eval outputs
- adapter artifacts
- Modal outputs and training artifacts
- local Gradio flagged data
- local traces
- local agent worktrees or scratch directories

The repository should not contain stray submission-risk files such as local `.env` files, raw transcript CSVs, JSONL eval outputs, adapter checkpoints, debug logs, or temporary experiment files outside ignored directories.

`requirements.txt` should stay generated from `uv` for Space compatibility. If dependencies change, regenerate it with `./export_space_requirements.sh` instead of editing it by hand. The README should match the actual torch pin currently written by the export script.

Dead code, commented-out experiments, and debug files should be removed only when they are clearly obsolete and unrelated to the spec-backed eval or training workflow. Do not remove active scripts under `evals/` that support the fine-tuning and adapter evaluation phases.

## Acceptance Criteria

- A submission-readiness spec exists under `spec/`.
- The README gives a two-line explanation of what the app does.
- The README prominently states about 3B total parameters across both models.
- Both model IDs are credited with links and Apache-2.0 license notes.
- The Cohere transcription model is clearly marked as gated, with duplication instructions for judges.
- The MiniCPM5 LoRA adapter is linked as `build-small-hackathon/minicpm5-speech-feedback-lora-v5`.
- The README states that the live Space runs the fine-tuned adapter.
- The README explains ZeroGPU quota behavior for anonymous versus logged-in Hugging Face users.
- The README includes the architecture flow from GPU transcription to CPU analysis to GPU review to scorecard.
- The README states this repo currently has no license file unless a license is added.
- The README includes the live Space link and a demo-video placeholder or final demo-video link.
- README Space front matter remains valid.
- Secret scans find no committed real secrets.
- `.gitignore` covers the project's private data, eval, adapter, Modal, trace, and local generated-output policy.
- `requirements.txt` is accurate for the current `uv.lock` and export script.
- Stray debug, token, transcript, eval-output, adapter, and temporary files are either removed from the submission tree or ignored by policy.

## Implementation Sketch

1. Add this phase spec.
2. Rewrite the README around submission needs while preserving valid Space YAML front matter.
3. Update `.gitignore` comments and patterns for private project policy gaps.
4. Run tracked and untracked secret/file scans.
5. Compare generated requirements output against `requirements.txt` without hand-editing generated content.
6. Remove or ignore only clearly local generated files and leave unrelated user changes intact.
