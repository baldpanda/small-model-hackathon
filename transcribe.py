from __future__ import annotations

import os
from functools import lru_cache

import soundfile as sf

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_RECORDING_SECONDS = 60
MODEL_SAMPLE_RATE = 16000
CHUNK_SECONDS = 15
CHUNK_SAMPLES = MODEL_SAMPLE_RATE * CHUNK_SECONDS


def _get_hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def _load_transcription_stack() -> tuple[object, object, object]:
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

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=token,
        trust_remote_code=True,
    )
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=token,
        trust_remote_code=True,
        device_map="auto",
    )
    return torch, processor, model


def _validate_duration(audio_path: str) -> float:
    try:
        info = sf.info(audio_path)
    except RuntimeError as exc:
        raise ValueError(f"Could not read the recording: {exc}") from exc

    duration_seconds = float(info.duration)
    if duration_seconds <= 0:
        raise ValueError("The recording appears to be empty. Try recording the speech again.")
    if duration_seconds > MAX_RECORDING_SECONDS:
        raise ValueError("The recording is longer than 60 seconds. Please keep the speech to one minute for now.")
    return duration_seconds


def _iter_audio_chunks(audio):
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk):
            yield chunk


def _transcribe_audio_chunk(torch, processor, model, audio, language: str) -> str:
    inputs = processor(
        audio=audio,
        sampling_rate=MODEL_SAMPLE_RATE,
        return_tensors="pt",
        language=language,
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    inputs.pop("length", None)
    inputs.pop("audio_chunk_index", None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)

    transcript = processor.decode(outputs, skip_special_tokens=True)
    if isinstance(transcript, list):
        transcript = transcript[0]

    return transcript.strip()


def transcribe_recording(audio_path: str, language: str = "en") -> str:
    _validate_duration(audio_path)
    torch, processor, model = _load_transcription_stack()
    try:
        from transformers.audio_utils import load_audio
    except ImportError as exc:
        raise RuntimeError("Transcription requires transformers audio utilities to load microphone audio.") from exc

    try:
        audio = load_audio(audio_path, sampling_rate=MODEL_SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(f"Failed to load the recording for transcription: {exc}") from exc

    transcripts = [
        _transcribe_audio_chunk(torch, processor, model, chunk, language)
        for chunk in _iter_audio_chunks(audio)
    ]
    text = " ".join(transcript for transcript in transcripts if transcript).strip()
    if not text:
        raise RuntimeError("The transcription model returned an empty transcript. Try recording again with clearer audio.")

    return text
