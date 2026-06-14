from __future__ import annotations

import unittest

from speech_stats import build_transcript_stats, classify_filler_rate, classify_wpm


class SpeechStatsTests(unittest.TestCase):
    def test_build_transcript_stats_computes_prompt_inputs(self) -> None:
        stats = build_transcript_stats(
            "Um, so this toast has ten plain spoken words now.",
            duration_seconds=30,
        )

        self.assertEqual(stats["word_count"], 10)
        self.assertEqual(stats["filler_count"], 2)
        self.assertEqual(stats["duration_mmss"], "0:30")
        self.assertEqual(stats["wpm"], 20.0)
        self.assertEqual(stats["wpm_band"], "slow (<120)")
        self.assertEqual(stats["filler_per_min"], 4.0)
        self.assertEqual(stats["filler_band"], "medium (2-4/min)")

    def test_classify_wpm_uses_rubric_buckets(self) -> None:
        self.assertEqual(classify_wpm(119.9), "slow (<120)")
        self.assertEqual(classify_wpm(120), "medium (120-180)")
        self.assertEqual(classify_wpm(180), "medium (120-180)")
        self.assertEqual(classify_wpm(180.1), "brisk (181-200)")
        self.assertEqual(classify_wpm(200), "brisk (181-200)")
        self.assertEqual(classify_wpm(200.1), "fast (>200)")

    def test_classify_filler_rate_uses_rubric_buckets(self) -> None:
        self.assertEqual(classify_filler_rate(1), "low (0-1/min)")
        self.assertEqual(classify_filler_rate(1.1), "medium (2-4/min)")
        self.assertEqual(classify_filler_rate(4), "medium (2-4/min)")
        self.assertEqual(classify_filler_rate(4.1), "high (5+/min)")


if __name__ == "__main__":
    unittest.main()
