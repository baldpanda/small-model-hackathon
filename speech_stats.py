from __future__ import annotations

from typing import Any

from filler_words import analyze_fillers
from timing import count_words


WPM_SLOW_LIMIT = 120
WPM_BRISK_LIMIT = 180
WPM_FAST_LIMIT = 200
FILLER_LOW_LIMIT = 1
FILLER_MEDIUM_LIMIT = 4


def build_transcript_stats(
    transcript: str,
    *,
    duration_seconds: float | None = None,
    duration_mmss: str | None = None,
) -> dict[str, Any]:
    word_count = count_words(transcript)
    filler_analysis = analyze_fillers(transcript)
    filler_count = filler_analysis.total_count

    stats: dict[str, Any] = {
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_counts": filler_analysis.counts,
        "notable_fillers": [
            {"filler": filler, "count": count}
            for filler, count in filler_analysis.notable_counts
        ],
    }

    if duration_seconds is not None:
        stats["duration_seconds"] = duration_seconds
        stats["duration_mmss"] = duration_mmss or format_duration_mmss(duration_seconds)

        if duration_seconds > 0:
            wpm = word_count / duration_seconds * 60
            filler_per_min = filler_count / duration_seconds * 60
            stats["wpm"] = round(wpm, 1)
            stats["wpm_band"] = classify_wpm(wpm)
            stats["filler_per_min"] = round(filler_per_min, 1)
            stats["filler_band"] = classify_filler_rate(filler_per_min)

    return stats


def classify_wpm(wpm: float) -> str:
    if wpm < WPM_SLOW_LIMIT:
        return "slow (<120)"
    if wpm <= WPM_BRISK_LIMIT:
        return "medium (120-180)"
    if wpm <= WPM_FAST_LIMIT:
        return "brisk (181-200)"
    return "fast (>200)"


def classify_filler_rate(filler_per_min: float) -> str:
    if filler_per_min <= FILLER_LOW_LIMIT:
        return "low (0-1/min)"
    if filler_per_min <= FILLER_MEDIUM_LIMIT:
        return "medium (2-4/min)"
    return "high (5+/min)"


def format_duration_mmss(duration_seconds: float) -> str:
    rounded_seconds = max(0, int(round(duration_seconds)))
    minutes, seconds = divmod(rounded_seconds, 60)
    return f"{minutes}:{seconds:02d}"
