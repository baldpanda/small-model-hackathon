# Phase 9 Trace Sharing Spec

## Goal

Prepare a Codex trace for hackathon sharing without committing raw trace data to the app repository.

## Scope

Phase 9 includes:

- a local `traces/raw/` directory for copied raw Codex trace files
- a local `traces/sanitized/` directory for redacted Codex trace files
- ignoring `traces/` in git because trace files can contain secrets and personal data
- a reusable script for producing a sanitized JSONL copy of a Codex trace
- default redaction of common token formats, private key blocks, bearer tokens, and local machine paths
- default removal of Codex reasoning items before public sharing

Phase 9 does not include:

- uploading traces to Hugging Face
- guaranteeing removal of every personal story, name, or wedding detail
- changing the runtime Gradio app
- adding new dependencies

## Required Behavior

- Raw trace files should stay local under `traces/raw/` and ignored by git.
- Sanitized trace files should be written under `traces/sanitized/` and ignored by git until manually reviewed.
- The sanitizer should read one JSON object per line and write one JSON object per line.
- Malformed JSONL lines should fail the script instead of being silently copied.
- High-confidence secret formats should be replaced with redaction placeholders.
- Local user and workspace paths should be replaced with stable placeholders.
- Reasoning trace items should be dropped by default.
- The script should support additional manual redaction terms for names or private details.

## Manual Review

Automated redaction is only a first pass. Before making a dataset public, manually inspect the sanitized trace for:

- real names
- wedding dates and locations
- venue details
- private stories
- copied private code or command output
- prompts or responses that should remain private
