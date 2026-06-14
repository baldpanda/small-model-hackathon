# Phase 11 Hugging Face Adapter Integration Spec

## Goal

Wire the trained MiniCPM5 speech-feedback LoRA adapter into the Hugging Face Space for an end-to-end app test.

This phase turns the Modal-trained adapter artifact into an app-loadable model path. It does not change the transcription stack, timing stats, filler stats, or scorecard prompt contract.

## Behavior

- The base model remains `openbmb/MiniCPM5-1B`.
- The tokenizer must always be loaded from the base model, not from the adapter.
- Thinking mode stays off via `enable_thinking=False`.
- The adapter is enabled only when `REVIEW_ADAPTER_ID` is set.
- If `REVIEW_ADAPTER_ID` is unset, the app uses the base model path.
- If `REVIEW_ADAPTER_ID` is set but PEFT or the adapter repo cannot load, startup should fail clearly rather than silently serving the wrong model.
- Private adapter repos should be accessed with the existing `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` secret.
- The output shape remains the fixed four-line contract:
  1. `Strength:`
  2. `Fix 1:`
  3. `Fix 2:`
  4. `Next run:`
- No retry or repair generation pass is added in this phase.

## Configuration

Expected Space environment:

```text
REVIEW_ADAPTER_ID=build-small-hackathon/minicpm5-speech-feedback-lora
HF_TOKEN=<secret if the adapter repo is private>
```

Use the actual uploaded adapter repo ID if it differs from the example above.

## Acceptance Criteria

- `peft` is included in the Space dependencies.
- `review_speech()` loads the adapter when `REVIEW_ADAPTER_ID` is configured.
- `review_speech()` keeps working with the base model when `REVIEW_ADAPTER_ID` is not configured.
- Adapter loading is logged without exposing secrets.
- The scorecard prompt and validator are unchanged except for adapter-aware model loading.
- Unit tests cover adapter env parsing and base-model fallback behavior.
