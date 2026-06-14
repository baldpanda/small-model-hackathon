from __future__ import annotations

import unittest

from evals.prepare_sft_dataset import (
    build_prompt_stats,
    normalize_gold_review,
    prepare_records,
    split_train_val,
)


class PrepareSftDatasetTests(unittest.TestCase):
    def test_normalize_gold_review_maps_training_labels(self) -> None:
        review = "\n".join(
            [
                "• Strength: Specific opening.",
                "• The one thing that matters: Choose one point.",
                "• Next step: Rehearse once.",
            ]
        )

        self.assertEqual(
            normalize_gold_review(review),
            "\n".join(
                [
                    "- Strength: Specific opening.",
                    "- Fix: Choose one point.",
                    "- Next run: Rehearse once.",
                ]
            ),
        )

    def test_build_prompt_stats_uses_gold_duration_and_stats_counts(self) -> None:
        stats = build_prompt_stats(
            {
                "computed_duration_mmss": "1:30",
                "computed_wpm": "160.0",
                "computed_wpm_band": "medium (120-180)",
                "computed_filler_per_min": "2.0",
                "computed_filler_band": "medium (2-4/min)",
            },
            {
                "computed_word_count": "240",
                "computed_filler_count": "3",
                "computed_notable_fillers": "um:2, so:1",
            },
        )

        self.assertEqual(stats["duration_seconds"], 90)
        self.assertEqual(stats["word_count"], 240)
        self.assertEqual(stats["wpm"], 160.0)
        self.assertEqual(stats["filler_count"], 3)
        self.assertEqual(stats["notable_fillers"], [{"filler": "um", "count": 2}, {"filler": "so", "count": 1}])

    def test_prepare_records_splits_and_keeps_smoke_in_train(self) -> None:
        gold_rows = {}
        stats_rows = {}
        for index in range(1, 17):
            row_id = str(index)
            speech_type = "best_man" if index <= 8 else "functional"
            quality = "clean_strong" if index % 2 else "weak"
            gold_rows[row_id] = {
                "id": row_id,
                "type": speech_type,
                "quality": quality,
                "variant": "test",
                "garble": "none",
                "computed_duration_mmss": "1:00",
                "computed_wpm": "120",
                "computed_wpm_band": "medium (120-180)",
                "computed_filler_per_min": "0",
                "computed_filler_band": "low (0-1/min)",
                "text": f"Transcript {index}",
                "gold_review": "• Strength: Clear.\n• Fix: Sharpen one point.\n• Next run: Rehearse once.",
            }
            stats_rows[row_id] = {
                "id": row_id,
                "computed_word_count": "20",
                "computed_filler_count": "0",
                "computed_notable_fillers": "",
                "computed_filler_counts_json": "{}",
            }

        records = prepare_records(
            gold_rows=gold_rows,
            stats_rows=stats_rows,
            eval_rows=[],
            seed=42,
            val_fraction=0.25,
            smoke_size=4,
        )

        self.assertEqual(len(records["train"]), 12)
        self.assertEqual(len(records["val"]), 4)
        self.assertEqual(len(records["smoke"]), 4)
        train_sources = {row["source_id"] for row in records["train"]}
        smoke_sources = {row["source_id"] for row in records["smoke"]}
        self.assertLessEqual(smoke_sources, train_sources)

    def test_split_train_val_uses_stratified_groups(self) -> None:
        rows = [
            {"source_id": str(index), "metadata": {"type": "best_man", "quality": "weak"}}
            for index in range(1, 11)
        ]

        train_rows, val_rows = split_train_val(rows, seed=42, val_fraction=0.2)

        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(val_rows), 2)


if __name__ == "__main__":
    unittest.main()
