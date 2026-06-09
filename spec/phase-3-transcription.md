# Phase 3 Transcription Spec

## Goal

Build the first audio-based rehearsal flow in the Gradio app.

Phase 3 should let a user record up to one minute of speech, transcribe that recording with `CohereLabs/cohere-transcribe-03-2026`, and feed the transcript into the existing echo response path so the app proves the full speech-to-text loop.

This phase is about replacing the manual text smoke test with a real spoken-input workflow, while keeping the output intentionally simple.

## Why This Comes Next

The current app proves deployment and a minimal interaction, but it does not yet exercise the actual rehearsal input mode described in the core product spec.

Adding a constrained transcription loop de-risks:

- microphone capture in Gradio
- model loading in the Hugging Face Space
- audio preprocessing and duration validation
- transcript handoff into downstream app logic

This supports the manifesto principle: get end-to-end early.

## User Story

As a best man rehearsing out loud, I want to record a short speech in the app and immediately see what the app heard, so I can confirm the transcription works before we add richer speech feedback.

## Scope

Phase 3 includes:

- microphone recording directly in the Gradio app
- a visible 60-second countdown during recording
- enforcement of a one-minute recording limit
- transcription with `CohereLabs/cohere-transcribe-03-2026`
- transcript display in the UI
- reuse of the existing echo response behavior, driven by the transcript text
- Hugging Face token-based access to the gated transcription model

Phase 3 does not include:

- audio file upload
- transcript editing before processing
- structure critique
- filler word analysis
- timing feedback beyond the one-minute limit
- multi-language UX work
- real-time or streaming transcription

## Required Behavior

The first audio flow should be intentionally narrow.

Required behavior:

- The user can start a microphone recording from the app.
- The UI shows a visible countdown for the 60-second limit.
- Recording stops accepting speech at 60 seconds.
- The app validates that usable audio was captured.
- The app transcribes the recording to text.
- The app shows the transcript to the user.
- The app passes the transcript into the same echo-style response path that currently handles typed text.
- The app shows the echo response in a separate output area.

If the recording cannot be transcribed, the app should return a clear, user-readable error instead of a blank result.

## Model and Runtime

Chosen transcription model:

- `CohereLabs/cohere-transcribe-03-2026`

Chosen runtime approach:

- Run the model inside the Hugging Face Space app process using the local Hugging Face model path rather than a separate transcription API.
- Use ZeroGPU as the execution environment for the Space.
- Authenticate gated model access with a Hugging Face token provided to the Space as a secret.

Implementation expectations:

- Load the model lazily and reuse it across requests where possible.
- Use `transformers` with any model-card-required options such as `trust_remote_code=True`.
- Normalize audio input into the format required by the model before inference.

## UI Shape

The app should move beyond the single `gr.Interface` smoke test and support a small multi-component layout.

Expected UI elements:

- app title and short description
- microphone recording input
- countdown or recording status text
- transcript output
- echo response output
- clear error/status messaging when recording or transcription fails

The first version should be one screen and avoid extra controls unless they are needed for the recording flow.

## Acceptance Criteria

- A user can record speech directly in the Gradio app.
- The UI shows a visible 60-second countdown or remaining-time indicator while recording.
- Audio beyond 60 seconds does not proceed as a normal transcription request.
- A valid short recording returns a readable transcript.
- The transcript is displayed in the interface.
- The existing echo behavior runs on the transcript text without changing its basic response style.
- The app handles empty, invalid, or failed recordings with a clear message.
- The flow works locally before deployment.
- The gated model can be loaded in the Space using the configured Hugging Face token.

## Open Questions

- Can Gradio hard-stop the browser recording cleanly at 60 seconds, or will the app need a UI countdown plus server-side rejection as the fallback?
- What exact dependency set from the model card is required for stable ZeroGPU execution?
- How much cold-start latency is acceptable for the first transcription request in the Space?

## Implementation Sketch

1. Replace the text-only input path with a small `gr.Blocks` layout for audio capture and outputs.
2. Add a transcription helper that loads and caches the Cohere model and processor.
3. Add audio preprocessing and duration validation for recorded input.
4. Add a flow function that transcribes audio and then calls the existing echo function with the transcript.
5. Add UI messaging for countdown, errors, transcript, and echo output.
6. Add the model and audio dependencies to the Python project.
7. Export `requirements.txt` from `uv` after dependency changes.
8. Validate locally with a short recording before deploying.

## Definition of Done

Phase 3 is done when a user can record a short speech in the app, see the transcript, and receive the existing echo response based on that transcript.

Once this is true, the next phase can add real speech-coaching analysis on top of a working transcription loop.
