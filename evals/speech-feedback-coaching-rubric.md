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
- 80-140 words by default
- one specific opening strength
- two prioritized fixes
- one concrete next rehearsal step

Gold examples should use exactly four short bullets for consistency. One tight paragraph is acceptable only for exploratory prompt tests, not for SFT targets or held-out scoring fixtures.

Scorecard order:

1. Earned strength.
2. Highest-priority fix.
3. Second fix.
4. Next rehearsal step.

Use a fifth bullet only when the input is unusually long or complex and a third fix is genuinely higher value than keeping the scorecard tight. For normal short rehearsals, more than two fixes is a shape failure.

## Taste Profile

The feedback should be direct, concise, and specific.

- Open with one real strength, not filler praise.
- Ground notes in what the speaker actually said.
- Preserve the speaker's voice, edge, and affectionate roast energy.
- Do not sanitize best-man humor into generic professional speaking.
- Protect the sincere emotional turn in wedding and celebration speeches.
- Match the feedback depth to the speech length and stakes.
- Pick the two fixes that matter most; use a third only for unusually long or complex inputs.
- Translate stats into actions, not recitations.
- Never invent transcript details, stats, emotions, or audience reactions.

Affectionate edge is good. Cruel, confusing, or alienating material should be redirected without flattening the whole speech.

## Scoring Protocol

Score each dimension as 0, 1, or 2. The short scale below is orientation only; the per-dimension anchors are authoritative.

- 0: absent, wrong, or generic for that dimension
- 1: partially right but vague, shallow, or inconsistent
- 2: specific, grounded, useful, and aligned with the taste profile

Reserve harmful or unusable behavior for the hard gates unless a dimension anchor explicitly captures it.

Use N/A only when a dimension is genuinely irrelevant to the speech type. Do not mark a dimension N/A merely because the transcript is short. Short speeches can still have context, structure, delivery, and next-step quality, just at a smaller scale.

Every N/A should include a one-line justification in eval notes.

For base vs. fine-tuned comparison, report per-dimension deltas as well as the total. The most important deltas are likely to appear in content specificity, voice and humor, emotional turn, delivery/stats translation, and no invented details.

Treat aggregate score as a secondary summary. Context tailoring, content specificity, and structural insight are intentionally correlated because transcript grounding is central to the product taste, so the total can overstate a single grounding improvement. Use per-dimension deltas and hard-gate rates as the main readout.

Default weighting is equal. If weighting is needed, use 1.5x for:

- content specificity
- voice and humor preservation
- emotional or occasion turn
- delivery/stats translation
- no invented details

## Hard Gates

Hard gates are output-level pass/fail checks. They answer whether the scorecard is usable at all. Dimension scores answer how good the scorecard is on specific qualities.

These failures should fail the whole output even if some dimensions score well:

- invents a story, relationship, venue, name, date, audience reaction, or stat
- exposes reasoning traces or thinking-mode content
- rewrites the full speech by default
- gives generic wedding advice to a clearly non-wedding transcript
- produces more than two fixes for a normal short rehearsal
- omits the concrete next rehearsal step
- ignores the stats block entirely when stats are present and relevant

Report hard-gate rate as a headline metric, separate from mean dimension score. For primary base-vs-fine-tuned comparison, compute mean dimension score on passing outputs only. Gated outputs may still be scored diagnostically, but those diagnostic scores should not be mixed into the primary quality mean.

Some gates overlap with dimensions on purpose. For example, invented details also score poorly on No Invented Details, and generic wedding advice may score poorly on Context Tailoring. The gate is the usability decision; the dimension score is the diagnostic explanation.

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
| 1 | Gives generic structure advice. | "Make sure you have a strong opening and closing." |
| 2 | Identifies the real arc and a concrete structural move. | "Open with the camping story, then let the cooking jokes lead into the sincere closing beat." |

Failure modes: intro-body-conclusion boilerplate, over-structuring an intentionally loose toast, missing multiple endings.

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

### 7. Proportionality and Restraint

Feedback matches the speech's length, stakes, and maturity.

| Score | Anchor | Example |
|---|---|---|
| 0 | Wildly mis-scaled. | Ten notes on a 40-second toast. |
| 1 | Reasonable length but unprioritized. | Five equal-weight issues with no ranking. |
| 2 | Two or three prioritized fixes scaled to the occasion. | "Two things matter: open with the camping story, then pause before the sincere turn." |

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

For evaluator calibration, keep separate non-training examples of weak and medium model outputs so scorers agree on what 0 and 1 look like.
