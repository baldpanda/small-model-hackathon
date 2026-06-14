from __future__ import annotations

import unittest

from evals.check_stats_schema_consistency import build_schema_report, extract_stats_block, stats_labels


class CheckStatsSchemaConsistencyTests(unittest.TestCase):
    def test_extract_stats_block_stops_before_transcript(self) -> None:
        user_content = "\n".join(
            [
                "Review this rehearsal.",
                "",
                "Stats:",
                "- Duration: 0:30 (30.0 seconds)",
                "- Word count: 80",
                "",
                "Transcript:",
                "Hello.",
            ]
        )

        self.assertEqual(extract_stats_block(user_content), "- Duration: 0:30 (30.0 seconds)\n- Word count: 80")

    def test_stats_labels_extracts_bullet_labels(self) -> None:
        labels = stats_labels(
            "\n".join(
                [
                    "- Duration: 0:30 (30.0 seconds)",
                    "- Word count: 80",
                    "- Pace: 160.0 wpm (medium (120-180))",
                    "- Fillers: 1 total, 2.0 per minute (medium (2-4/min))",
                ]
            )
        )

        self.assertEqual(labels, ["Duration", "Word count", "Pace", "Fillers"])

    def test_build_schema_report_tracks_missing_core_labels(self) -> None:
        report = build_schema_report(
            [
                {
                    "id": "row-1",
                    "stats_block": "\n".join(
                        [
                            "- Duration: 0:30 (30.0 seconds)",
                            "- Word count: 80",
                            "- Pace: 160.0 wpm (medium (120-180))",
                            "- Fillers: 1 total, 2.0 per minute (medium (2-4/min))",
                        ]
                    ),
                },
                {"id": "row-2", "stats_block": "- Word count: 80\n- Fillers: 0 total"},
            ]
        )

        self.assertEqual(report["count"], 2)
        self.assertEqual(report["missing_core"]["row-2"], ["Duration", "Pace"])


if __name__ == "__main__":
    unittest.main()
