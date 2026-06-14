# Phase 11 Hugging Face Adapter Integration Spec

## Goal

Wire the trained MiniCPM5 speech-feedback LoRA adapter into the Hugging Face Space for an end-to-end app test.

This phase turns the Modal-trained adapter artifact into an app-loadable model path. It does not change the transcription stack, timing stats, or filler stats. The current app-facing scorecard contract is documented below because the adapter has to run behind that contract.

## Behavior

- The base model remains `openbmb/MiniCPM5-1B`.
- The tokenizer must always be loaded from the base model, not from the adapter.
- Thinking mode stays off via `enable_thinking=False`.
- The adapter is enabled only when `REVIEW_ADAPTER_ID` is set.
- If `REVIEW_ADAPTER_ID` is unset, the app uses the base model path.
- If `REVIEW_ADAPTER_ID` is set but PEFT or the adapter repo cannot load, startup should fail clearly rather than silently serving the wrong model.
- Private adapter repos should be accessed with the existing `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` secret.
- Populated prompt logging is opt-in via `REVIEW_LOG_PROMPT=1` and should be enabled only while debugging because it logs transcript text and stats.
- The output shape follows the app prompt contract.
- Substantive clips use the four-line contract:
  1. `Strength:`
  2. `Fix 1:`
  3. `Fix 2:`
  4. `Next run:`
- Clips under 50 words or under 20 seconds use the shorter three-line contract:
  1. `Strength:`
  2. `Fix:`
  3. `Next run:`
- No retry or repair generation pass is added in this phase.

## Configuration

Expected Space environment:

```text
REVIEW_ADAPTER_ID=build-small-hackathon/minicpm5-speech-feedback-lora
HF_TOKEN=<secret if the adapter repo is private>
```

Use the actual uploaded adapter repo ID if it differs from the example above.

Temporary prompt debugging:

```text
REVIEW_LOG_PROMPT=1
```

When enabled, Space logs include the populated MiniCPM chat-template prompt that is sent for tokenization. Turn it off after debugging because transcripts may contain private user speech.

## Acceptance Criteria

- `peft` is included in the Space dependencies.
- `review_speech()` loads the adapter when `REVIEW_ADAPTER_ID` is configured.
- `review_speech()` keeps working with the base model when `REVIEW_ADAPTER_ID` is not configured.
- Adapter loading is logged without exposing secrets.
- Prompt logging is disabled by default and logs the populated prompt only when `REVIEW_LOG_PROMPT` is truthy.
- The scorecard prompt and validator accept the short three-line contract for clips with too little material for two useful fixes.
- Unit tests cover adapter env parsing and base-model fallback behavior.
