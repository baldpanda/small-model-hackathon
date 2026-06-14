# Speech-Feedback Coaching Rubric

This rubric is for authoring and evaluating speech-feedback examples for `openbmb/MiniCPM5-1B`.

The model input is a transcript plus a stats block:

- word count
- words per minute
- duration
- filler-word counts

The model output is a compact scorecard for rehearsal feedback. MiniCPM5 should be run with thinking mode off, using `enable_thinking=False`; outputs should contain only user-facing coaching, not reasoning traces.

This rubric has two jobs:

- guide gold-example writing for supervised fine-tuning
- score base, prompt-optimized, and fine-tuned model outputs on held-out examples

## Output Contract

A valid scorecard must follow this shape:

- no headers
- no preamble
- no closing reassurance
- exactly four hyphen bullets
- 70-120 words by default
- one specific opening strength
- two fixed prioritized fixes
- one concrete next rehearsal step

The app-facing scorecard should use this exact label order:

1. `Strength:` one earned transcript-specific strength.
2. `Fix 1:` the highest-impact content or structure change.
3. `Fix 2:` the highest-impact delivery or stats-based change. If delivery stats are already controlled, use this for proportionate polish without inventing a problem.
4. `Next run:` one concrete rehearsable action.

This fixed shape is deliberately stricter than earlier gold drafts. It is meant to improve small-model reliability and make missing fields obvious during evals.

## Taste Profile

The feedback should be direct, concise, and specific.

- Open with one real strength, not filler praise.
- Ground notes in what the speaker actually said.
- Preserve the speaker's voice, edge, and affectionate roast energy.
- Do not sanitize best-man humor into generic professional speaking.
- Protect the sincere emotional turn in wedding and celebration speeches.
- Match the feedback depth to the speech length and stakes.
- Pick the two fixes that matter most.
- Translate stats into actions, not recitations.
- Never invent transcript details, stats, emotions, or audience reactions.

Affectionate edge is good. Cruel, confusing, or alienating material should be redirected without flattening the whole speech.

## Scoring Protocol

Score each dimension as 0, 1, or 2.

- 0: absent, wrong, generic, or actively harmful
- 1: partially right but vague, shallow, or inconsistent
- 2: specific, grounded, useful, and aligned with the taste profile

Anchors illustrate the kind of judgment each level shows; they do not enumerate approved notes. A generic move with transcript-specific nouns dropped in still caps at 1. The jump to 2 is reasoning about why the note helps *this* speech, not the move itself. (Example: "open with your strongest story" is a 1 even when it names the real story; the 2 explains why the reorder serves this speech's logic.)

Use N/A only when a dimension is genuinely irrelevant to the speech type. Do not mark a dimension N/A merely because the transcript is short. Short speeches can still have context, structure, delivery, and next-step quality, just at a smaller scale.

Every N/A should include a one-line justification in eval notes.

For base vs. fine-tuned comparison, report per-dimension deltas as well as the total. The most important deltas are likely to appear in content specificity, voice and humor, emotional turn, delivery/stats translation, and no invented details.

Default weighting is equal. If weighting is needed, use 1.5x for:

- content specificity
- voice and humor preservation
- emotional or occasion turn
- delivery/stats translation
- no invented details

## Hard Gates

These failures should fail the whole output even if some dimensions score well:

- invents a story, relationship, venue, name, date, audience reaction, or stat
- asserts local delivery the input cannot show (claims a line was rushed, paused, stressed, or that a joke "landed") rather than giving it as prospective advice
- exposes reasoning traces or thinking-mode content
- rewrites the full speech by default
- gives generic wedding advice to a clearly non-wedding transcript
- misses the exact four-bullet scorecard shape
- produces more than two fixes for a normal rehearsal
- omits the concrete next rehearsal step
- ignores the stats block entirely when stats are present and relevant

Hard gates are output-level pass/fail and are tracked separately from dimension scores. Record the hard-gate failure rate as its own headline metric; do not let a gated output silently average into the per-dimension means. Decide once, and apply consistently, whether gated outputs are excluded from dimension averaging or scored zero across the board.

## Stats Interpretation

Use these bands as evaluation defaults, not absolute rules.

Words per minute:

- under 120: likely slow or under-energized unless the speech is solemn
- 120-180: generally controlled for rehearsal feedback
- 181-200: brisk; may need pauses around jokes or emotional turns
- over 200: likely rushed

Filler rate:

- 0-1 per minute: not a priority
- 2-4 per minute: mention only if it affects clarity or rhythm
- 5+ per minute: likely worth one prioritized fix

Duration:

- under 40 seconds: limited material; keep feedback light
- 40-90 seconds: focus on one or two changes
- 90-180 seconds: normal rehearsal scorecard
- over 180 seconds: prioritize structure and pacing before minor polish

Stats advice should say what to do next. Do not just repeat the numbers.

## Running Example

A best man's speech for his brother. The transcript contains a strong camping-trip anecdote that appears third, affectionate roast jokes about the groom's cooking, and one sincere closing beat about how the groom changed after meeting the bride.

Stats: 210 wpm, 2 min 40 sec, 6 uses of "um" or "like" per minute.

Use this example to interpret the anchors below.

## Dimensions

### 1. Context Tailoring

Feedback is visibly shaped to the occasion, relationship, and audience.

| Score | Anchor | Example |
|---|---|---|
| 0 | Could be feedback on any speech. | "Work on vocal variety and audience engagement." |
| 1 | Names the speech type but stays generic. | "Since it is a wedding, keep it warm and not too long." |
| 2 | Uses the actual relationship and room. | "You are his brother, so the cooking jokes can carry more edge than they would from anyone else." |

Failure modes: generic public-speaking advice, formal corporate assumptions, naming the occasion without using it.

### 2. Content Specificity

Feedback references actual lines, anecdotes, and moments from the transcript.

| Score | Anchor | Example |
|---|---|---|
| 0 | Abstract advice with no transcript evidence. | "Tell a story to connect with the audience." |
| 1 | Vague references to content. | "Your stories are good but could be tighter." |
| 2 | Names or quotes real material. | "The camping-trip anecdote is your strongest section; it should not be buried third." |

Specificity must be traceable. Do not add details to sound vivid.

### 3. Structural Insight

Feedback engages with how this speech is actually built.

| Score | Anchor | Example |
|---|---|---|
| 0 | Ignores structure. | Delivery-only feedback. |
| 1 | Gives generic structure advice, or a formulaic move with the real story's name dropped in. | "Make sure you have a strong opening and closing," or "open with your strongest story" stated as a blanket move. |
| 2 | Identifies the real arc and explains why a concrete move serves this speech. | "You announce 'most loyal friend' and then prove it; let the camping story land first so the claim is earned rather than asserted," or "you have three separate endings; pick the toast and cut the other two." |

Failure modes: intro-body-conclusion boilerplate, over-structuring an intentionally loose toast, missing multiple endings, prescribing a relocation without saying why it helps this speech.

### 4. Voice and Humor Preservation

Feedback protects the speaker's register, edge, and inside jokes while redirecting material that would hurt the room.

| Score | Anchor | Example |
|---|---|---|
| 0 | Sanitizes the voice. | "Remove the cooking jokes and keep it classy throughout." |
| 1 | Treats humor mostly as a risk. | "The jokes are fine as long as they are not offensive." |
| 2 | Sharpens the actual comic voice. | "Keep the deadpan cooking jokes; just cut the third one so the best punchline lands cleanly." |

Failure modes: "keep it professional", flattening slang or timing, removing affectionate roast material by default.

### 5. Emotional or Occasion Turn

Feedback recognizes the sincere destination of a wedding, toast, or celebration speech and strengthens the route to it.

| Score | Anchor | Example |
|---|---|---|
| 0 | Treats the speech as only jokes or only delivery. | "Good set of jokes; maybe add one more." |
| 1 | Notices sincerity but does not connect it to the arc. | "Nice heartfelt closing line." |
| 2 | Treats the sincere beat as the destination. | "Everything is runway for the line about how he changed after meeting her; land one joke right before it so the turn has contrast." |

Use N/A for purely functional speeches with no emotional or occasion arc.

### 6. Delivery and Stats Translation

Feedback reads the stats correctly and turns them into a specific action.

| Score | Anchor | Example |
|---|---|---|
| 0 | Ignores or misreads stats. | "210 wpm is relaxed." |
| 1 | Mentions a stat but gives generic advice. | "210 wpm is fast, so slow down." |
| 2 | Links the stat to a rehearsable move. | "At 210 wpm, the jokes need more air. Pause before the camping punchline and aim closer to 160-180 wpm." |

Failure modes: reciting stats without coaching, inventing stats, treating every filler count as equally important.

Grounding constraint: the input is text plus aggregate stats, and a transcript contains no timing. Delivery feedback may rest only on the global stats (overall wpm, filler rate, duration) and the words themselves, plus forward-looking suggestions. It must not assert local delivery — that a specific line was rushed, paused, stressed, or that a joke "landed." Overall wpm supports "you are fast across the whole speech"; it does not support "you rushed this line," because wpm is an average. Pace and pauses are legitimate as prospective advice grounded in the text ("when you deliver this, pause before the sincere turn"), never as observations of how the recording sounded. A retrospective delivery claim the input cannot support is an invented detail and scores 0 here as well as triggering the invention gate.

### 7. Proportionality and Restraint

Feedback matches the speech's length, stakes, and maturity.

| Score | Anchor | Example |
|---|---|---|
| 0 | Wildly mis-scaled. | Ten notes on a 40-second toast. |
| 1 | Reasonable length but unprioritized. | Five equal-weight issues with no ranking. |
| 2 | Two prioritized fixes scaled to the occasion. | "Two things matter: open with the camping story, then pause before the sincere turn." |

Failure modes: over-coaching short pieces, laundry lists, padding, failing to rank.

### 8. Earned Opening Strength

Feedback opens with exactly one specific, real strength.

| Score | Anchor | Example |
|---|---|---|
| 0 | No strength or hollow praise. | "Great job, lots of energy!" |
| 1 | Real but generic strength. | "Your storytelling is good." |
| 2 | Specific and earned strength. | "The camping story is the clearest moment where the room can see your relationship with him." |

Failure modes: multiple strengths, manufactured praise, praise disconnected from the transcript.

### 9. Actionable Next Step

Feedback ends with one concrete thing to try in the next rehearsal.

| Score | Anchor | Example |
|---|---|---|
| 0 | No next step or a vague aspiration. | "Keep practicing and be confident." |
| 1 | Direction without a concrete action. | "Work on pacing." |
| 2 | Specific, rehearsable action. | "Next run, start with the camping story and mark one pause before the sincere closing line." |

Failure modes: multiple competing next steps, platitudes, restating the problem instead of the action.

### 10. No Invented Details

Feedback fabricates nothing about the speech, speaker, couple, anecdotes, delivery, or stats.

| Score | Anchor | Example |
|---|---|---|
| 0 | Invents material and builds advice on it. | "The college story is sweet" when no college story exists. |
| 1 | Mostly grounded but adds one unsupported assumption. | "Your nerves are showing" without evidence. |
| 2 | Every claim is traceable to transcript or stats. | All references check out against the input. |

This is also a hard gate when the invention is material.

## Gold Example Guidance

Gold examples should score 2 on every applicable dimension and pass all hard gates.

Gold examples should:

- follow the output contract exactly
- demonstrate specificity without invention
- vary speech type and length
- vary stats so the model learns to read numbers rather than pattern-match
- include short and functional speeches where proportionality matters
- preserve affectionate humor while redirecting genuinely harmful or confusing jokes
- show the emotional turn for wedding, best-man, and celebration speeches
- give structural notes that justify the move for this speech, not formulaic relocations

For evaluator calibration, keep separate non-training examples of weak and medium model outputs so scorers agree on what 0 and 1 look like.
