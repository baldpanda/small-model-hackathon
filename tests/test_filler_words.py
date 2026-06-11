from __future__ import annotations

import unittest

from filler_words import analyze_fillers, format_filler_summary


class FillerWordTests(unittest.TestCase):
    def test_counts_single_word_fillers_case_insensitively(self) -> None:
        analysis = analyze_fillers("Um, actually, LIKE, right, so this is basically ready.")

        self.assertEqual(analysis.counts["um"], 1)
        self.assertEqual(analysis.counts["actually"], 1)
        self.assertEqual(analysis.counts["like"], 1)
        self.assertEqual(analysis.counts["right"], 1)
        self.assertEqual(analysis.counts["so"], 1)
        self.assertEqual(analysis.counts["basically"], 1)

    def test_counts_phrase_fillers_before_single_words(self) -> None:
        analysis = analyze_fillers("Sort of, kind of, you know. So we begin.")

        self.assertEqual(analysis.counts["sort of"], 1)
        self.assertEqual(analysis.counts["kind of"], 1)
        self.assertEqual(analysis.counts["you know"], 1)
        self.assertEqual(analysis.counts["so"], 1)

    def test_phrase_matching_handles_punctuation_between_words(self) -> None:
        analysis = analyze_fillers("You, know, I kind-of mean it is sort...of working.")

        self.assertEqual(analysis.counts["you know"], 1)
        self.assertEqual(analysis.counts["kind of"], 1)
        self.assertEqual(analysis.counts["sort of"], 1)

    def test_whole_word_matching_avoids_embedded_terms(self) -> None:
        analysis = analyze_fillers("The likely answer is upright and sober.")

        self.assertEqual(analysis.counts["like"], 0)
        self.assertEqual(analysis.counts["right"], 0)
        self.assertEqual(analysis.counts["so"], 0)

    def test_summary_omits_zero_count_noise(self) -> None:
        summary = format_filler_summary(analyze_fillers("This speech starts cleanly."))

        self.assertIn("No notable filler words", summary)
        self.assertNotIn("um:", summary)

    def test_summary_shows_notable_counts(self) -> None:
        summary = format_filler_summary(analyze_fillers("Um, um, you know, so."))

        self.assertIn("Tracked fillers found: 4", summary)
        self.assertIn("- um: 2 times", summary)
        self.assertIn("- so: 1 time", summary)
        self.assertIn("- you know: 1 time", summary)


if __name__ == "__main__":
    unittest.main()
