from __future__ import annotations

import sys
import types
import unittest

sys.modules.setdefault("soundfile", types.SimpleNamespace())

import timing
from timing import analyze_timing, classify_pace, count_words, format_timing_summary, get_audio_duration_seconds


class _FakeAudioInfo:
    def __init__(self, duration: float) -> None:
        self.duration = duration


class TimingTests(unittest.TestCase):
    def _stub_duration(self, duration_seconds: float) -> None:
        original_info = getattr(timing.sf, "info", None)
        timing.sf.info = lambda audio_path: _FakeAudioInfo(duration_seconds)
        if original_info is None:
            self.addCleanup(lambda: delattr(timing.sf, "info"))
        else:
            self.addCleanup(lambda: setattr(timing.sf, "info", original_info))

    def test_count_words_handles_punctuation_and_contractions(self) -> None:
        transcript = "I'm here, we're ready -- best-man speech time."

        self.assertEqual(count_words(transcript), 8)

    def test_get_audio_duration_seconds_reads_recording_length(self) -> None:
        self._stub_duration(0.5)

        self.assertAlmostEqual(get_audio_duration_seconds("recording.wav"), 0.5, places=2)

    def test_analyze_timing_calculates_words_per_minute(self) -> None:
        self._stub_duration(1.0)
        transcript = "one two"

        analysis = analyze_timing("recording.wav", transcript)

        self.assertEqual(analysis.word_count, 2)
        self.assertAlmostEqual(analysis.words_per_minute, 120.0, places=1)
        self.assertEqual(analysis.pace_label, "steady")

    def test_classify_pace_uses_expected_thresholds(self) -> None:
        self.assertEqual(classify_pace(109.9), "slow/spacious")
        self.assertEqual(classify_pace(110), "steady")
        self.assertEqual(classify_pace(165), "steady")
        self.assertEqual(classify_pace(165.1), "fast")

    def test_format_timing_summary_marks_short_samples_low_confidence(self) -> None:
        self._stub_duration(1.0)

        summary = format_timing_summary(analyze_timing("recording.wav", "one two"))

        self.assertIn("Confidence: low", summary)
        self.assertIn("Try this next:", summary)


if __name__ == "__main__":
    unittest.main()
