# Phase 6 Wedding Scorecard Polish Spec

## Goal

Make the working rehearsal app feel deliberately custom instead of like a default Gradio form.

This phase should improve the first impression, scanability, and hackathon presentation while preserving the current end-to-end loop: record a rehearsal, transcribe it, review it, and show timing and filler feedback.

## Visual Direction

The app should use a "Wedding Scorecard" direction: a best man rehearsal report styled like a wedding program crossed with a judging card.

The interface should feel warm, specific, and off-brand for Gradio without adding a separate frontend framework. Visual cues can include print-style rules, scorecard panels, invitation-inspired borders, and ceremony accent colors.

The palette should avoid becoming a single beige or brown theme. Use ivory, ink, deep green, burgundy, and gold accents so the app looks distinctive while staying readable.

## Scope

Phase 6 includes:

- custom Gradio CSS
- a stronger first-viewport header
- a more polished recording and status area
- scorecard-style result sections
- clearer visual separation between transcript, speech feedback, timing feedback, and filler feedback
- preserving the existing countdown and processing behavior

Phase 6 does not include:

- changes to the transcription model
- changes to the review model
- new dependencies
- a custom JavaScript frontend
- new required inputs
- generated image assets
- deployment workflow changes

## Required Behavior

The app remains a single Gradio screen.

Required behavior:

- The user can record up to one minute of rehearsal audio.
- The countdown/status behavior remains visible and usable.
- The Review speech button still calls the existing processing function.
- The transcript, speech feedback, timing feedback, filler feedback, and status all remain visible after processing.
- Model, timing, and filler failure behavior remains unchanged.
- The UI should look substantially different from default Gradio components.
- The layout should remain readable on desktop and mobile widths.

### Rehearsal Booth recording controls

- The Record control is visually distinct from the Review speech button — a different colour family and silhouette — so the input action and the commit action do not compete for attention.
- Post-recording editing controls (waveform trim, download, clear-X) are hidden. The user either commits the take with Review speech or starts over with Try again; there is no in-between editing step.
- A Try again button sits alongside Review speech. Activating it clears the current recording and resets the transcript, speech feedback, timing feedback, filler feedback, and status panels to their initial placeholder state so the user can record again immediately.
- Try again must also work after recording has stopped but before Review speech is clicked: the waveform should clear, the recording timer/status should return to its initial state, and Review speech should become disabled until a new recording exists.

## Acceptance Criteria

- Existing timing and filler tests still pass.
- `uv run python app.py` starts the app locally.
- The first screen clearly presents the app as a best man speech rehearsal tool.
- The recording area and result sections use custom styling instead of default Gradio form presentation.
- Feedback outputs are easier to scan than the previous stacked textbox layout.
- No dependency export is required because dependencies are unchanged.
