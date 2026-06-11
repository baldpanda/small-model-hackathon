# Phase 5 Timing And Filler Analysis Spec

## Goal

Add deterministic rehearsal metrics on top of the current transcription and small-model review loop.

Phase 5 should show the speaker how long they spoke, how fast they spoke, and which filler words or repeated verbal habits stood out. The feedback should be practical, lightweight, and separate from the MiniCPM review so users can distinguish measured facts from model-generated coaching.

This phase is about completing more of the original core loop without adding another model or widening the interface too much.

## Why This Comes Next

Phase 4 turns a recorded rehearsal into a transcript and useful speech feedback. The next missing pieces from the phase 1 core loop are timing and filler-word feedback.

Adding these as deterministic helpers de-risks:

- audio duration reporting outside the transcription validation path
- estimated words-per-minute calculation
- filler-word counting from transcripts
- clear presentation of measured feedback alongside model feedback
- useful coaching even when the model feedback is brief

This supports the manifesto principles: polish the core loop, keep the human voice, and use focused workflows rather than relying only on model output.

## User Story

As someone rehearsing out loud, I want to know whether I am speaking too quickly and which filler habits are showing up, so I can practise the next run with one or two concrete delivery goals.

## Scope

Phase 5 includes:

- recording duration in seconds
- estimated word count from the transcript
- estimated words per minute
- simple pacing feedback based on the words-per-minute estimate
- deterministic filler-word and phrase counts from the transcript
- a concise filler summary that highlights only notable habits
- UI updates to display timing and filler feedback separately from the transcript and model review
- focused tests for timing and filler helper logic

Phase 5 does not include:

- target speech length input
- comparison against a user-selected target duration
- longer recording support beyond the current 60-second limit
- real-time pacing or filler alerts while speaking
- audio-level disfluency detection
- transcript editing before analysis
- changes to the transcription or review models
- another scoring system or overall grade
- hackathon submission polish such as demo videos or README restructuring

## Required Behavior

The flow should remain one screen and build directly on the current app.

Required behavior:

- The user records a short rehearsal.
- The app transcribes the recording with the existing transcription path.
- The app reviews the transcript with the existing MiniCPM review path.
- The app calculates deterministic timing feedback from the recording duration and transcript.
- The app calculates deterministic filler feedback from the transcript.
- The app shows transcript, speech feedback, timing feedback, filler feedback, and status in clearly separated outputs.
- Timing and filler feedback should still be useful for very short transcripts, but should say when there is too little material to judge confidently.
- If transcription fails, the existing transcription error behavior should remain unchanged.
- If model review fails after transcription succeeds, the app should still show the transcript and should still attempt to show timing and filler feedback.
- If timing or filler analysis fails unexpectedly, the app should return a clear user-readable message instead of hiding the transcript or review.

## Timing Analysis

Timing analysis should use the recorded audio duration and transcript word count.

Expected metrics:

- total duration, rounded to one decimal place or nearest second
- estimated word count
- estimated words per minute
- short pacing label
- one practical pacing suggestion

Initial pacing bands:

- Under 110 words per minute: likely slow or spacious.
- 110 to 165 words per minute: likely steady.
- Over 165 words per minute: likely fast.

These bands are intentionally simple. They should guide practice, not diagnose delivery perfectly. The app should avoid overconfident timing advice when the recording is only a few seconds long or has very few words.

Phase 5 should not compare the rehearsal against a target length. The current 60-second limit is a product constraint for this phase, not the user's target speech length.

## Filler Word Analysis

Filler analysis should run on the transcript text and count common words or phrases.

Initial fillers to track:

- um
- uh
- like
- you know
- sort of
- kind of
- basically
- actually
- literally
- right
- so

The implementation should count phrase fillers such as `you know`, `sort of`, and `kind of` without double-counting their component words as separate fillers.

The user-facing summary should not dump every zero count. It should highlight the most frequent habits and give one practical rehearsal suggestion. If there are no notable fillers, the app should say that clearly and encourage the user to focus on pacing or structure instead.

The tone should stay supportive and non-shaming. The point is to help the speaker notice habits, not make the rehearsal feel like an exam.

## UI Shape

The current phase 4 layout should remain recognizable.

Expected UI elements:

- app title and short description
- microphone recording input
- recording countdown or status text
- transcript output
- speech feedback output
- timing feedback output
- filler feedback output
- status output

The timing and filler outputs can be textboxes or markdown-style text, whichever fits the existing Gradio layout best. The interface should avoid extra controls in this phase.

## Acceptance Criteria

- A valid recording returns a readable transcript.
- The app returns MiniCPM-generated speech feedback as in phase 4.
- The app shows total recording duration.
- The app shows estimated word count and words per minute.
- The app labels the estimated pace using the initial pacing bands.
- The app counts the defined filler words and phrases from the transcript.
- The filler summary highlights notable habits without showing noisy zero-count output.
- Phrase fillers are not double-counted as individual component words.
- Very short transcripts receive proportionate timing and filler feedback.
- If review generation fails, transcript, timing, and filler outputs remain available when possible.
- The feature uses deterministic helper logic and does not add another model.
- The flow works locally before deployment.

## Open Questions

- Are the initial pacing bands good enough for one-minute rehearsals, or should they be adjusted after trying real recordings?
- Should `so` and `right` count only in obvious filler positions, or is a simple whole-word count acceptable for the first pass?
- Should the timing and filler summaries eventually be merged into the MiniCPM prompt so the model can reference measured metrics?
- Should a later phase add a target speech length once recordings longer than one minute are supported?

## Implementation Sketch

1. Add a `timing.py` helper for audio duration, word count, words per minute, pacing labels, and timing-summary formatting.
2. Add a `filler_words.py` helper for deterministic filler counting and filler-summary formatting.
3. Reuse `soundfile` for audio duration so no new dependency is required.
4. Update `app.py` so `process_rehearsal` returns timing and filler outputs in addition to transcript, speech feedback, and status.
5. Keep transcript visible when downstream review, timing, or filler formatting fails.
6. Add focused tests under `tests/` for timing and filler helpers.
7. Validate the Gradio flow locally with `uv run python app.py`.

## Definition Of Done

Phase 5 is done when a user can record a short rehearsal, see the transcript and MiniCPM review, and also get clear deterministic feedback about pace and filler habits.

Once this is true, a later phase can focus on hackathon polish, target-length controls, longer rehearsal support, or folding measured metrics into richer model coaching.
