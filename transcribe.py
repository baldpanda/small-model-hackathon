from __future__ import annotations

import os
import logging
import time
from functools import lru_cache

import soundfile as sf

from rehearsal_limits import validate_recording_duration_seconds

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MODEL_SAMPLE_RATE = 16000
MAX_TRANSCRIPTION_TOKENS = 512

LOGGER = logging.getLogger(__name__)


def _get_hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def _load_transcription_stack() -> tuple[object, object, object]:
    load_started_at = time.perf_counter()
    try:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Transcription dependencies are missing. Install the project dependencies before running phase 3."
        ) from exc

    token = _get_hugging_face_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to access the gated Cohere model."
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=token,
    )
    runtime_device = _place_model_for_runtime(torch, model)
    model.eval()
    LOGGER.info(
        "Loaded transcription stack on %s in %.1fs",
        runtime_device,
        time.perf_counter() - load_started_at,
    )
    return torch, processor, model


def _place_model_for_runtime(torch: object, model: object) -> str:
    if torch.cuda.is_available():
        model.to("cuda")
        return "cuda"

    try:
        model.to("cuda")
    except (AssertionError, RuntimeError) as exc:
        LOGGER.info("CUDA startup placement unavailable; using CPU: %s", exc)
        model.to("cpu")
        return "cpu"

    return "cuda"


_TRANSCRIPTION_STACK = _load_transcription_stack()


def _validate_duration(audio_path: str, duration_seconds: float | None = None) -> float:
    if duration_seconds is None:
        try:
            info = sf.info(audio_path)
        except RuntimeError as exc:
            raise ValueError(f"Could not read the recording: {exc}") from exc
        duration_seconds = float(info.duration)

    validate_recording_duration_seconds(duration_seconds)
    return duration_seconds


def transcribe_recording(audio_path: str, language: str = "en", duration_seconds: float | None = None) -> str:
    _validate_duration(audio_path, duration_seconds)
    torch, processor, model = _TRANSCRIPTION_STACK
    try:
        from transformers.audio_utils import load_audio
    except ImportError as exc:
        raise RuntimeError("Transcription requires transformers audio utilities to load microphone audio.") from exc

    try:
        audio = load_audio(audio_path, sampling_rate=MODEL_SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(f"Failed to load the recording for transcription: {exc}") from exc

    inputs = processor(
        audio=audio,
        sampling_rate=MODEL_SAMPLE_RATE,
        language=language,
    )
    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)
    inputs.pop("length", None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TRANSCRIPTION_TOKENS)

    if audio_chunk_index is None:
        transcript = processor.decode(outputs, skip_special_tokens=True)
    else:
        transcript = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
    if isinstance(transcript, list):
        transcript = transcript[0]

    text = transcript.strip()
    if not text:
        raise RuntimeError("The transcription model returned an empty transcript. Try recording again with clearer audio.")

    return text
