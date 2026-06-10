# Phase 4 Small-Model Review Spec

## Goal

Replace the phase 3 echo response with useful best man speech feedback generated from the transcript.

Phase 4 should take the transcript produced by the current recording flow, review it with `openbmb/MiniCPM5-1B`, and return concise, practical feedback that helps the speaker improve the next rehearsal.

This phase is about turning the working transcription loop into the first genuinely useful speech-coaching loop while keeping the scope narrow.

## Why This Comes Next

Phase 3 proves that the app can capture speech, transcribe it, and hand the transcript into downstream logic.

The next highest-value step is to replace the placeholder echo with model-backed review. This de-risks:

- loading a second small model in the Space
- prompt design for best man speech feedback
- transcript handoff from transcription to review
- user-facing error handling when review generation fails
- sponsor-award fit for OpenBMB models

This supports the manifesto principle: use small models, focused prompts, and clear workflows to provide practical feedback without flattening the speaker's voice.

## User Story

As a best man rehearsing out loud, I want the app to review what I said and tell me what to improve next, so I can practise without needing another person to listen every time.

## Scope

Phase 4 includes:

- using the existing transcript output as the review input
- local small-model review with `openbmb/MiniCPM5-1B`
- best man speech feedback focused on structure, clarity, warmth, audience fit, and next rehearsal priorities
- a clear feedback output in the Gradio interface
- user-readable errors when review generation fails
- keeping the model combination within the hackathon's 32B total model-size limit

Phase 4 does not include:

- filler-word counts
- duration, words-per-minute, or target-length feedback
- transcript editing before review
- full speech rewriting
- NVIDIA Nemotron integration
- llama.cpp runtime integration
- video, image, or multimodal review
- real-time feedback while speaking

## Required Behavior

The review flow should stay one-screen and build directly on the current audio workflow.

Required behavior:

- The user records a short rehearsal.
- The app transcribes the recording with the existing transcription path.
- The app sends the transcript to the review model.
- The app shows the transcript and generated speech feedback.
- The feedback is specific to best man speeches.
- The feedback avoids rewriting the whole speech by default.
- The feedback gives the speaker concrete things to practise next.

If transcription fails, the existing transcription error behavior should remain unchanged. If review generation fails after transcription succeeds, the app should still show the transcript and return a clear feedback error instead of a blank response.

## Model and Runtime

Chosen review model:

- `openbmb/MiniCPM5-1B`
- Model card: https://huggingface.co/openbmb/MiniCPM5-1B

Chosen runtime approach:

- Run the review model inside the Hugging Face Space app process.
- Load the model lazily and reuse it across requests where possible.
- Use the existing local-model deployment approach rather than an external inference API.
- Keep the app compatible with Hugging Face Spaces and ZeroGPU.

Implementation expectations:

- Use `transformers` with the model's required tokenizer and causal language model classes.
- Use the model's chat template where available.
- Disable thinking mode when supported, such as with `enable_thinking=False`, so the user sees direct feedback rather than reasoning traces.
- Keep generation bounded so feedback remains concise and latency stays acceptable.

Dependency compatibility:

- The current project pins `transformers>=5.4.0,<5.5.0` and `torch` through `uv` to preserve the known-working Cohere transcription stack.
- The MiniCPM5 model card recommends `transformers>=5.6`, which conflicts with the current upper bound.
- Phase 4 implementation upgrades `transformers` to `>=5.6.0,<5.7.0` so MiniCPM5 can use its documented Transformers path.
- Before merging, re-test Cohere transcription with the upgraded Transformers stack and use the Phase 4 model canary to prove both models can run in one process.
- Regenerate `requirements.txt` through `./export_space_requirements.sh` after dependency changes.

This model choice helps the project target OpenBMB sponsor-award eligibility and the broader small-model spirit of the hackathon.

## Feedback Shape

The review should be readable as a short coaching note, not a scorecard.

Expected sections:

- Overall impression
- What is working
- What to improve
- Next rehearsal checklist

The feedback should be supportive, direct, and specific. It should preserve the speaker's own voice, personal stories, and natural humour. It should avoid generic wedding-speech advice that could apply to any transcript.

## UI Shape

The current phase 3 layout should remain mostly intact.

Expected UI changes:

- Update the app description so it says the app transcribes and reviews the speech.
- Rename the response output from `App response` to `Speech feedback`.
- Update the feedback placeholder so it no longer refers to echo behavior.
- Keep transcript, feedback, and status as separate outputs.

The first review version should avoid additional controls unless they are required for model operation.

## Acceptance Criteria

- A valid recording returns a readable transcript.
- The app returns model-generated feedback from `openbmb/MiniCPM5-1B`.
- The feedback is tailored to best man speech rehearsal.
- The feedback includes strengths, improvements, and a short next rehearsal checklist.
- The feedback does not rewrite the full speech by default.
- If review generation fails, the transcript remains visible and the app returns a clear user-readable error.
- The app remains Gradio-hosted and local-model-based.
- The model combination stays within the hackathon's 32B model-size limit.
- The flow works locally before deployment.

## Open Questions

- Does `openbmb/MiniCPM5-1B` run cleanly with the currently pinned `transformers` version?
- Is the combined transcription and review latency acceptable under the current Space GPU duration?
- Does the Space need additional memory or model-loading safeguards once both models are cached?
- Should a later phase use the MiniCPM GGUF variant to target the llama.cpp bonus badge?

## Implementation Sketch

1. Add a `review.py` helper that loads and caches `openbmb/MiniCPM5-1B`.
2. Add a `review_speech(transcript: str) -> str` function with a fixed best man speech feedback prompt.
3. Replace the phase 3 echo response call in `app.py` with the review helper.
4. Keep transcript output visible even if review generation fails.
5. Update UI copy and output labels from echo language to review language.
6. Update dependencies only if MiniCPM5 requires changes to the current project pins.
7. Regenerate `requirements.txt` from `uv` if dependency files change.
8. Validate with a direct review smoke test and a full Gradio recording flow.

## Definition of Done

Phase 4 is done when a user can record a short rehearsal, see the transcript, and receive concise OpenBMB-generated feedback that gives them clear priorities for the next practice run.

Once this is true, a later phase can add deterministic timing and filler-word analysis on top of the review loop.
