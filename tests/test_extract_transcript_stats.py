from __future__ import annotations

import unittest

from evals.extract_transcript_stats import flatten_stats, metadata_for_output


class ExtractTranscriptStatsTests(unittest.TestCase):
    def test_metadata_omits_text_by_default(self) -> None:
        row = {"id": "1", "type": "toast", "text": "private speech", "transcript": "private"}

        self.assertEqual(metadata_for_output(row, include_text=False), {"id": "1", "type": "toast"})

    def test_flatten_stats_formats_notable_fillers(self) -> None:
        flattened = flatten_stats(
            {
                "word_count": 12,
                "filler_count": 2,
                "filler_counts": {"um": 2},
                "notable_fillers": [{"filler": "um", "count": 2}],
            }
        )

        self.assertEqual(flattened["computed_word_count"], 12)
        self.assertEqual(flattened["computed_filler_count"], 2)
        self.assertEqual(flattened["computed_notable_fillers"], "um:2")
        self.assertEqual(flattened["computed_filler_counts_json"], '{"um": 2}')


if __name__ == "__main__":
    unittest.main()
