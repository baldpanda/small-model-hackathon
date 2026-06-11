from __future__ import annotations

import re
from dataclasses import dataclass

import soundfile as sf


SLOW_WPM_LIMIT = 110
FAST_WPM_LIMIT = 165
LOW_CONFIDENCE_SECONDS = 5
LOW_CONFIDENCE_WORDS = 8
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class TimingAnalysis:
    duration_seconds: float
    word_count: int
    words_per_minute: float
    pace_label: str
    low_confidence: bool


def get_audio_duration_seconds(audio_path: str) -> float:
    try:
        info = sf.info(audio_path)
    except RuntimeError as exc:
        raise ValueError(f"Could not read the recording duration: {exc}") from exc

    duration_seconds = float(info.duration)
    if duration_seconds <= 0:
        raise ValueError("The recording appears to be empty, so timing feedback is unavailable.")
    return duration_seconds


def count_words(transcript: str) -> int:
    return len(WORD_PATTERN.findall(transcript))


def classify_pace(words_per_minute: float) -> str:
    if words_per_minute < SLOW_WPM_LIMIT:
        return "slow/spacious"
    if words_per_minute <= FAST_WPM_LIMIT:
        return "steady"
    return "fast"


def analyze_timing(audio_path: str, transcript: str) -> TimingAnalysis:
    duration_seconds = get_audio_duration_seconds(audio_path)
    word_count = count_words(transcript)
    words_per_minute = 0.0
    if word_count:
        words_per_minute = word_count / duration_seconds * 60

    return TimingAnalysis(
        duration_seconds=duration_seconds,
        word_count=word_count,
        words_per_minute=words_per_minute,
        pace_label=classify_pace(words_per_minute),
        low_confidence=duration_seconds < LOW_CONFIDENCE_SECONDS or word_count < LOW_CONFIDENCE_WORDS,
    )


def _pacing_suggestion(analysis: TimingAnalysis) -> str:
    if analysis.low_confidence:
        return "Try a slightly longer rehearsal clip before treating the pace estimate as reliable."
    if analysis.pace_label == "slow/spacious":
        return "Keep the breathing room, but check that pauses feel intentional rather than hesitant."
    if analysis.pace_label == "fast":
        return "Slow the setup lines and leave space after jokes or emotional beats."
    return "This is a workable rehearsal pace; keep using pauses at transitions."


def format_timing_summary(analysis: TimingAnalysis) -> str:
    lines = [
        f"Duration: {analysis.duration_seconds:.1f} seconds",
        f"Estimated words: {analysis.word_count}",
        f"Estimated pace: {analysis.words_per_minute:.1f} words per minute ({analysis.pace_label})",
    ]
    if analysis.low_confidence:
        lines.append("Confidence: low because this clip is very short or has very few words.")
    lines.append(f"Try this next: {_pacing_suggestion(analysis)}")
    return "\n".join(lines)


def summarize_timing(audio_path: str, transcript: str) -> str:
    return format_timing_summary(analyze_timing(audio_path, transcript))
