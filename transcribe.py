from __future__ import annotations

import os
from functools import lru_cache

import soundfile as sf

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_RECORDING_SECONDS = 60
MODEL_SAMPLE_RATE = 16000


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

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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
        torch_dtype=torch_dtype,
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


def transcribe_recording(audio_path: str, language: str = "en") -> str:
    _validate_duration(audio_path)
    torch, processor, model = _load_transcription_stack()
    try:
        audio, sample_rate = sf.read(audio_path, always_2d=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load the recording for transcription: {exc}") from exc

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if sample_rate != MODEL_SAMPLE_RATE:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("Transcription requires librosa to resample microphone audio to 16 kHz.") from exc

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=MODEL_SAMPLE_RATE,
        )
        sample_rate = MODEL_SAMPLE_RATE

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language=language,
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    inputs.pop("length", None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)

    transcript = processor.decode(outputs, skip_special_tokens=True)
    if isinstance(transcript, list):
        transcript = transcript[0]

    text = transcript.strip()
    if not text:
        raise RuntimeError("The transcription model returned an empty transcript. Try recording again with clearer audio.")

    return text
