from __future__ import annotations

import html
import re
from dataclasses import dataclass
from types import SimpleNamespace

SLOW_WPM_LIMIT = 110
FAST_WPM_LIMIT = 165
LOW_CONFIDENCE_SECONDS = 5
LOW_CONFIDENCE_WORDS = 8
GAUGE_MIN_WPM = 60.0
GAUGE_MAX_WPM = 220.0
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
sf = SimpleNamespace()


@dataclass(frozen=True)
class TimingAnalysis:
    duration_seconds: float
    word_count: int
    words_per_minute: float
    pace_label: str
    low_confidence: bool


def get_audio_duration_seconds(audio_path: str) -> float:
    if not hasattr(sf, "info"):
        import soundfile as soundfile_module

        sf.info = soundfile_module.info

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


def _format_clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _gauge_pct(wpm: float) -> float:
    pct = (wpm - GAUGE_MIN_WPM) / (GAUGE_MAX_WPM - GAUGE_MIN_WPM) * 100.0
    return max(0.0, min(100.0, pct))


def format_pacing_preview_html(duration_seconds: float) -> str:
    """Shown after recording, before the transcript exists."""
    return (
        '<div class="pace-stats">'
        f'<div class="pace-stat"><div class="num">{_format_clock(duration_seconds)}</div>'
        '<div class="lbl">duration</div></div>'
        '<div class="pace-stat"><div class="num">&mdash;</div><div class="lbl">words</div></div>'
        '<div class="pace-stat"><div class="num">&mdash;</div><div class="lbl">wpm</div></div>'
        '</div>'
        '<p class="pace-note">Pacing lands once the transcript is ready.</p>'
    )


def format_pacing_html(audio_path: str, transcript: str) -> str:
    analysis = analyze_timing(audio_path, transcript)
    wpm = analysis.words_per_minute
    in_band = SLOW_WPM_LIMIT <= wpm <= FAST_WPM_LIMIT
    flag = "" if (in_band or wpm == 0) else " num--flag"

    band_left = _gauge_pct(SLOW_WPM_LIMIT)
    band_width = _gauge_pct(FAST_WPM_LIMIT) - band_left
    marker = (
        f'<span class="marker" style="left:{_gauge_pct(wpm):.1f}%"></span>' if wpm else ""
    )

    confidence = (
        '<p class="pace-note"><strong>Low confidence:</strong> very short clip or few words.</p>'
        if analysis.low_confidence else ""
    )

    return (
        '<div class="pace-stats">'
        f'<div class="pace-stat"><div class="num">{_format_clock(analysis.duration_seconds)}</div>'
        '<div class="lbl">duration</div></div>'
        f'<div class="pace-stat"><div class="num">{analysis.word_count}</div>'
        '<div class="lbl">words</div></div>'
        f'<div class="pace-stat"><div class="num{flag}">{wpm:.0f}</div>'
        '<div class="lbl">wpm</div></div>'
        '</div>'
        '<div class="pace-gauge">'
        f'<span class="band" style="left:{band_left:.1f}%;width:{band_width:.1f}%"></span>'
        f'{marker}'
        '</div>'
        '<div class="pace-scale"><span>60</span>'
        '<span style="color:#183d34">steady 110\u2013165</span><span>220</span></div>'
        f'<p class="pace-note">{html.escape(_pacing_suggestion(analysis))}</p>'
        f'{confidence}'
    )
