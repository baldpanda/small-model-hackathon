from __future__ import annotations

import argparse
import logging
import os


COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MINICPM_MODEL_ID = "openbmb/MiniCPM5-1B"
SAMPLE_RATE = 16000
LOGGER = logging.getLogger("phase4_models")


def get_hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    return None


def model_kwargs(token: str | None) -> dict[str, str]:
    if token:
        return {"token": token}
    return {}


def check_imports() -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CohereAsrForConditionalGeneration

    LOGGER.info("torch=%s", torch.__version__)
    LOGGER.info("transformers=%s", transformers.__version__)
    LOGGER.info("AutoModelForCausalLM=%s", AutoModelForCausalLM.__name__)
    LOGGER.info("AutoProcessor=%s", AutoProcessor.__name__)
    LOGGER.info("AutoTokenizer=%s", AutoTokenizer.__name__)
    LOGGER.info("CohereAsrForConditionalGeneration=%s", CohereAsrForConditionalGeneration.__name__)


def load_minicpm(token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading %s", MINICPM_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MINICPM_MODEL_ID, **model_kwargs(token))
    model = AutoModelForCausalLM.from_pretrained(
        MINICPM_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        **model_kwargs(token),
    )
    messages = [
        {
            "role": "user",
            "content": "Reply in one short sentence: MiniCPM is ready.",
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("MiniCPM generated an empty response.")

    LOGGER.info("MiniCPM output: %s", text)
    return tokenizer, model


def load_cohere(token: str | None):
    if not token:
        raise RuntimeError("Cohere Transcribe is gated. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN before running this check.")

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from transformers.audio_utils import load_audio

    LOGGER.info("Loading %s", COHERE_MODEL_ID)
    processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID, token=token)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        COHERE_MODEL_ID,
        token=token,
        device_map="auto",
    )
    audio_file = hf_hub_download(
        repo_id=COHERE_MODEL_ID,
        filename="demo/voxpopuli_test_en_demo.wav",
        token=token,
    )
    audio = load_audio(audio_file, sampling_rate=SAMPLE_RATE)
    inputs = processor(
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        language="en",
    )
    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)
    inputs.pop("length", None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)

    if audio_chunk_index is None:
        transcript = processor.decode(outputs, skip_special_tokens=True)
    else:
        transcript = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language="en",
        )
    if isinstance(transcript, list):
        transcript = transcript[0]

    text = transcript.strip()
    if not text:
        raise RuntimeError("Cohere Transcribe returned an empty transcript.")

    LOGGER.info("Cohere transcript: %s", text)
    return processor, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prove Phase 4 model dependency compatibility.")
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="Only verify imports and installed versions.",
    )
    parser.add_argument(
        "--skip-minicpm",
        action="store_true",
        help="Skip loading and generating with MiniCPM5.",
    )
    parser.add_argument(
        "--skip-cohere",
        action="store_true",
        help="Skip loading and transcribing with Cohere Transcribe.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings and errors.",
    )
    return parser.parse_args()


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.quiet)
    token = get_hugging_face_token()

    check_imports()
    if args.imports_only:
        LOGGER.info("Imports-only check complete.")
        return

    cohere_stack = None
    minicpm_stack = None
    if not args.skip_cohere:
        cohere_stack = load_cohere(token)
    if not args.skip_minicpm:
        minicpm_stack = load_minicpm(token)

    if cohere_stack and minicpm_stack:
        LOGGER.info("Both Phase 4 models loaded and generated/transcribed in one Python process.")
    else:
        LOGGER.info("Selected Phase 4 model checks completed.")


if __name__ == "__main__":
    main()
