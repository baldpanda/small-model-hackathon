# Phase 7 Runtime Observability And ZeroGPU Fit Spec

## Goal

Reduce avoidable ZeroGPU time and make processing latency visible enough to guide the next optimisation pass.

The app should show how long each processing step takes for a rehearsal run, and model stacks should be initialised during Space startup rather than on the first anonymous GPU request.

## Why This Comes Next

The working app now does all core tasks: transcription, speech review, timing analysis, and filler analysis. On ZeroGPU, anonymous users have limited daily GPU quota, so the next risk is not feature coverage but runtime fit.

The current implementation caches model loaders, but those loaders are first called from inside the GPU-decorated request handler. That means the first user request can spend GPU lease time loading model weights. Hugging Face ZeroGPU guidance recommends placing models on `cuda` at root module startup so CUDA placement is optimised before user requests.

## Scope

Phase 7 includes:

- module-level initialisation of the transcription stack
- module-level initialisation of the review stack
- per-action timing for transcription, deterministic timing analysis, filler analysis, review generation, formatting, and total request time
- a visible processing-timing report in the status output
- progressive UI updates after validation, transcription, deterministic metrics, and final review
- concise server logs for the same timing data
- a shorter dynamic ZeroGPU duration than the previous fixed 120-second request

Phase 7 does not include:

- changing model choices
- adding another model
- changing the one-minute recording limit
- changing prompt intent or review output structure
- replacing Gradio with a custom frontend
- adding external telemetry or analytics

## Required Behavior

- Importing the app should initialise both model stacks once.
- The request handler should reuse already loaded model objects.
- A valid rehearsal should still return transcript, speech feedback, timing feedback, filler feedback, and status.
- A valid rehearsal should stream partial results instead of waiting for all processing to finish before updating the UI.
- The app should show recording duration before model inference when the audio can be read.
- The app should show the transcript as soon as transcription completes.
- The app should show timing and filler feedback before review generation completes.
- The status output should include a compact timing report after successful processing.
- If review generation fails after transcription succeeds, the status output should still include the timings collected before failure.
- If transcription fails, the status output should include timings collected before the failure when possible.
- Deterministic timing and filler analysis should remain available when review generation fails.
- Timing instrumentation should not add meaningful overhead or new dependencies.
- Local non-CUDA development should still work where model dependencies and credentials are available.

## Timing Report

The report should track these labels:

- transcription
- timing analysis
- filler analysis
- review generation
- formatting
- total

The report can be plain Markdown. It should be compact enough to live in the existing status panel without adding another required control.

## ZeroGPU Duration

The request duration should be reduced from the previous 120 seconds after startup loading removes model-load work from the request path.

The initial maximum target duration is 45 seconds. The app should estimate duration from the uploaded audio length, add a small overhead for transcription and review generation, clamp quick clips to a 15-second minimum, and clamp one-minute clips to a 45-second maximum. This reflects observed 23-second and roughly 50-second rehearsals both completing under 6 seconds once model loading moved out of the request path. The timing report should be used to decide whether the maximum can be lowered again later.

## Acceptance Criteria

- Model stack creation is no longer first triggered from inside `process_rehearsal`.
- The GPU-decorated request handler has a duration lower than 120 seconds.
- A successful run includes a processing timing report in the status output.
- A successful run progressively updates the status, transcript, metrics, and final feedback outputs.
- Review failure paths preserve transcript, deterministic feedback, and collected timings.
- Transcription failure paths preserve a clear user-readable error and collected timings.
- Existing timing and filler tests still pass.
- `python -m py_compile` succeeds for changed Python modules.
