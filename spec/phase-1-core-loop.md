# Phase 1 Core Loop Spec

## Goal

Build the first end-to-end speech practice loop:

1. The user records or uploads a spoken best man speech rehearsal.
2. The app transcribes the audio to text.
3. The app reviews the speech structure.
4. The app reviews timing and pacing.
5. The app highlights filler words and repeated verbal habits.
6. The user gets practical feedback they can use before the next rehearsal.

This phase is about proving the smallest useful version of the product. It should be good enough for real use by the intended group of friends, even if the interface and feedback are still simple.

## Intended User

A best man preparing a wedding speech who wants to practise out loud and get feedback without needing another person to listen every time.

The app should feel like a friendly rehearsal partner: honest, specific, and supportive.

## Input

The first version should accept an audio recording of a speech rehearsal.

Possible input paths:

- Record directly in the Gradio app.
- Upload an existing audio file.

The app should also allow optional context:

- Speaker name.
- Couple names.
- Target speech length.
- Desired tone, such as warm, funny, sentimental, or light roast.
- Any sections the speaker wants to include.

Optional context can come later if needed. The minimum viable input is audio.

## Processing

### Speech-to-Text

Use a voice-to-text model to create a transcript from the rehearsal audio.

The transcript should preserve enough detail for feedback, including:

- Approximate wording.
- Obvious pauses or breaks when available.
- Repeated filler words.
- Disfluent phrasing where the transcription model captures it.

The transcription model counts toward the hackathon's total 32B model size limit.

### Structure Review

Use a small language model to review the transcript for best man speech structure.

The review should look for:

- A clear opening.
- How the speaker introduces themselves.
- Whether the speech has a beginning, middle, and ending.
- Whether stories are easy to follow.
- Whether jokes or anecdotes connect back to the couple.
- Whether the tone feels appropriate for a wedding audience.
- Whether the ending lands with warmth and clarity.

The model should not rewrite the whole speech by default. It should focus on feedback and concrete suggestions.

### Timing Review

Measure the rehearsal duration from the audio input and compare it with the target speech length when provided.

The timing review should include:

- Total rehearsal duration.
- Estimated words per minute.
- Whether the speech feels too short, too long, or close to target.
- Practical pacing advice, such as where to slow down, trim, or add breathing room.

If no target duration is provided, the app should still report the actual duration and words per minute.

### Filler Word Review

Detect and summarize filler words and repeated verbal habits.

Initial filler words to track:

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

The app should distinguish between ordinary uses and distracting repetition where possible. A simple count is acceptable for the first version, but the feedback should avoid shaming the speaker.

## Output

The first version should return:

- Transcript: the speech text produced from the audio.
- Structure summary: a short assessment of the speech shape.
- Timing summary: duration, estimated pace, and target-length feedback.
- Strengths: what is already working.
- Improvements: specific changes to try next.
- Filler word summary: counts and notable repeated habits.
- Next rehearsal checklist: 3 to 5 practical things to focus on.

The output should be written in a supportive tone. The user should finish reading it knowing what to practise next.

## Acceptance Criteria

- A user can provide audio in the Gradio app.
- The app returns a readable transcript.
- The app identifies at least the most common filler words from the transcript.
- The app gives structure feedback tailored to best man speeches.
- The app reports rehearsal duration and estimated words per minute.
- If the user provides a target length, the app compares the rehearsal against it.
- The app produces a short next-rehearsal checklist.
- The flow works locally before deployment.
- The chosen model combination stays within the total 32B parameter limit.

## Non-Goals

- Perfect transcription accuracy.
- Real-time coaching while speaking.
- Full speech rewriting.
- Video analysis.
- Sentiment analysis of the wedding audience.
- A strict scoring system that makes the app feel like an exam.

## Open Questions

- Should the first version use direct recording, file upload, or both?
- Should filler words be counted from the raw transcript only, or should the app also ask the review model to identify verbal habits?
- Should the structure review use a fixed rubric or a more conversational critique?
- What should the default target speech length be if the user does not provide one?
- Should timing feedback be based only on total duration, or should it eventually identify sections that drag or feel rushed?
- Which voice-to-text model and review model give the best balance of quality, latency, simplicity, and the 32B total model limit?

## First Implementation Sketch

- `app.py`: Gradio app with audio input and optional text context.
- `transcribe.py`: speech-to-text helper.
- `review.py`: best man speech review prompt and model call.
- `filler_words.py`: deterministic filler word counting from transcript text.
- `timing.py`: audio duration and words-per-minute helper.
- `requirements.txt`: Gradio and model/client dependencies.

Keep this first version boring in the best way: one screen, one recording, one useful feedback report.
