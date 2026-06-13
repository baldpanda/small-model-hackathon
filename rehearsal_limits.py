from __future__ import annotations

import math


MIN_RECORDING_SECONDS = 10
MAX_RECORDING_SECONDS = 120
MIN_GPU_DURATION_SECONDS = 15
MAX_GPU_DURATION_SECONDS = 30
GPU_DURATION_BASE_SECONDS = 12
GPU_DURATION_AUDIO_RATIO = 0.15

SHORT_RECORDING_MESSAGE = "That was a little short. Ready when you are to practise the speech."


def format_duration_label(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes and remaining_seconds:
        return f"{minutes} minute {remaining_seconds} seconds"
    if minutes == 1:
        return "1 minute"
    if minutes:
        return f"{minutes} minutes"
    if seconds == 1:
        return "1 second"
    return f"{seconds} seconds"


def accepted_recording_window_label() -> str:
    return f"{format_duration_label(MIN_RECORDING_SECONDS)} to {format_duration_label(MAX_RECORDING_SECONDS)}"


def validate_recording_duration_seconds(duration_seconds: float) -> None:
    if duration_seconds <= 0:
        raise ValueError("The recording appears to be empty. Try recording the speech again.")
    if duration_seconds < MIN_RECORDING_SECONDS:
        raise ValueError(SHORT_RECORDING_MESSAGE)
    if duration_seconds > MAX_RECORDING_SECONDS:
        raise ValueError(
            f"The recording is longer than {format_duration_label(MAX_RECORDING_SECONDS)}. "
            "Please keep this rehearsal to two minutes for now."
        )


def estimate_gpu_duration_seconds(duration_seconds: float) -> int:
    estimated_seconds = math.ceil(GPU_DURATION_BASE_SECONDS + duration_seconds * GPU_DURATION_AUDIO_RATIO)
    return min(MAX_GPU_DURATION_SECONDS, max(MIN_GPU_DURATION_SECONDS, estimated_seconds))
