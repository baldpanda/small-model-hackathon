# Phase 8 Two-Minute Rehearsal Pilot Spec

## Goal

Pilot a longer rehearsal window without making anonymous ZeroGPU usage too expensive.

Phase 8 should let a best man practise a more realistic speech section by increasing the recording limit from one minute to two minutes. It should also reject clips that are too short to be useful before requesting GPU time, so accidental taps and tiny samples do not spend anonymous user quota.

## Why This Comes Next

The current app can transcribe, review, measure pace, count filler words, stream progress, and report processing timings. The remaining product tension is rehearsal length.

One minute is useful for quick retries, but it is short for practising a best man speech. At the same time, Hugging Face ZeroGPU gives unauthenticated users a small daily quota, so the app should not request a GPU lease based on the full recording length when observed processing time is much shorter.

The latest observed run took about 1.8 seconds for transcription and 3.8 seconds for review generation on a roughly 45-second speech. That supports a cautious two-minute pilot with a capped dynamic GPU budget rather than a jump to full-speech support.

## Scope

Phase 8 includes:

- increasing the maximum recording length from 60 seconds to 120 seconds
- adding a 10-second minimum recording length
- blocking too-short recordings before any GPU-decorated work
- showing an encouraging message for too-short recordings
- centralising recording and GPU budget limits so UI copy, validation, and runtime duration stay aligned
- reducing the maximum dynamic ZeroGPU duration from 45 seconds to 30 seconds for the pilot
- showing the requested GPU budget in status output and server logs
- increasing transcription output headroom for longer transcripts
- focused tests for recording-window validation and GPU budget calculation

Phase 8 does not include:

- full-speech-length recording support
- target speech length input
- model changes
- prompt redesign
- new dependencies
- external telemetry
- queue or quota analytics beyond the existing status timings

## Required Behavior

- The user can record up to two minutes of rehearsal audio.
- The browser countdown and app copy should say two minutes, not one minute.
- When the browser countdown reaches two minutes, it should click the active audio Stop control before resetting countdown state so the recording does not continue past the pilot window.
- Recordings under 10 seconds should not call the GPU-decorated processing path.
- Too-short recordings should return a friendly message such as: "That was a little short. Ready when you are to practise the speech."
- Recordings over 120 seconds should not call the GPU-decorated processing path.
- Valid recordings from 10 to 120 seconds should continue through the existing transcript, timing, filler, and speech review flow.
- The app should still show recording duration before model inference when duration can be read.
- If duration cannot be read, the app should return a clear user-readable message rather than guessing and requesting GPU time.
- The current progressive updates should remain: validation, transcript, deterministic metrics, then review.
- Existing failure behavior should remain: transcription failures are clear, and review failures preserve transcript and deterministic feedback when possible.

## Recording Window

The intended pilot window is:

- minimum recording duration: 10 seconds
- maximum recording duration: 120 seconds

The minimum is a quality and quota guard, not a judgement on the speaker. It should prevent accidental short clips and very small samples from consuming ZeroGPU time.

The product should still leave room for practising an opening. Ten seconds is intentionally low enough for a short intro, but high enough that the review has some material to work with.

## ZeroGPU Duration

The GPU duration should be based on expected processing time rather than speech length.

Initial pilot limits:

- minimum GPU duration request: 15 seconds
- maximum GPU duration request: 30 seconds
- duration estimate: `ceil(12 + audio_duration_seconds * 0.15)`, clamped to the 15-to-30-second range

Examples:

- a 10-second clip requests 15 seconds
- a 45-second clip requests about 19 seconds
- a 120-second clip requests 30 seconds

The app should log and display the requested GPU duration alongside actual processing timings. If real two-minute runs regularly approach the 30-second cap, the next phase should tune the cap or reduce model work before increasing the recording limit again.

## Implementation Sketch

1. Add a small shared limits module for recording duration and GPU duration constants.
2. Move duration validation into a non-GPU wrapper that runs before the GPU-decorated processing path.
3. Keep the GPU-decorated path focused on transcription, timing, filler analysis, review generation, formatting, and timing collection for already-valid audio.
4. Update `transcribe.py` validation to use the shared 120-second maximum.
5. Update the JavaScript countdown, status copy, and error messages to use the shared recording window.
6. Increase transcription generation headroom from `256` to `512` tokens for longer two-minute transcripts.
7. Add pure helper tests for duration validation and GPU budget calculation.

## Acceptance Criteria

- A recording under 10 seconds returns the encouraging short-clip message and does not request GPU work.
- A recording over 120 seconds returns a clear limit message and does not request GPU work.
- A valid 10-to-120-second recording follows the existing end-to-end flow.
- The UI countdown and visible app copy describe a two-minute limit.
- The browser timeout path stops the active recording before clearing its saved audio button reference.
- The status output includes actual processing timings and the requested GPU budget.
- The dynamic GPU duration is clamped between 15 and 30 seconds.
- The current timing and filler tests still pass.
- New validation and GPU-budget tests pass.
- `python -m py_compile` succeeds for changed Python modules.

## Open Questions

- After trying real two-minute best-man rehearsals, is the 30-second GPU cap still comfortably above actual processing time?
- Should a later phase add target speech length once two-minute support proves stable?
- Should short opening-line practice eventually have a separate lightweight mode that skips model review?
