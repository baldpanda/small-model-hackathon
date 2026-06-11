from __future__ import annotations

import re
from dataclasses import dataclass


PHRASE_FILLERS = ("you know", "sort of", "kind of")
SINGLE_WORD_FILLERS = (
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "literally",
    "right",
    "so",
)
FILLERS = PHRASE_FILLERS + SINGLE_WORD_FILLERS


@dataclass(frozen=True)
class FillerAnalysis:
    counts: dict[str, int]
    total_count: int

    @property
    def notable_counts(self) -> list[tuple[str, int]]:
        return sorted(
            ((filler, count) for filler, count in self.counts.items() if count > 0),
            key=lambda item: (-item[1], item[0]),
        )


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in phrase.split()]
    return re.compile(r"\b" + r"\W+".join(words) + r"\b", re.IGNORECASE)


def _word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _blank_matches(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    characters = list(text)
    count = 0
    for match in pattern.finditer(text):
        count += 1
        for index in range(match.start(), match.end()):
            characters[index] = " "
    return "".join(characters), count


def analyze_fillers(transcript: str) -> FillerAnalysis:
    remaining_text = transcript.lower()
    counts = dict.fromkeys(FILLERS, 0)

    for phrase in PHRASE_FILLERS:
        remaining_text, counts[phrase] = _blank_matches(remaining_text, _phrase_pattern(phrase))

    for word in SINGLE_WORD_FILLERS:
        matches = _word_pattern(word).findall(remaining_text)
        counts[word] = len(matches)

    return FillerAnalysis(counts=counts, total_count=sum(counts.values()))


def _format_count(filler: str, count: int) -> str:
    if count == 1:
        return f"{filler}: 1 time"
    return f"{filler}: {count} times"


def format_filler_summary(analysis: FillerAnalysis) -> str:
    notable_counts = analysis.notable_counts
    if not notable_counts:
        return (
            "No notable filler words detected from the tracked list.\n"
            "Try this next: focus on pacing, structure, or clearer transitions."
        )

    shown_counts = notable_counts[:5]
    lines = [
        f"Tracked fillers found: {analysis.total_count}",
        "Most noticeable habits:",
    ]
    lines.extend(f"- {_format_count(filler, count)}" for filler, count in shown_counts)

    top_filler, top_count = notable_counts[0]
    if top_count == 1:
        suggestion = "There is no major repeated filler pattern yet; keep rehearsing naturally."
    else:
        suggestion = f"Try replacing repeated '{top_filler}' moments with a short pause."
    lines.append(f"Try this next: {suggestion}")
    return "\n".join(lines)


def summarize_fillers(transcript: str) -> str:
    return format_filler_summary(analyze_fillers(transcript))
