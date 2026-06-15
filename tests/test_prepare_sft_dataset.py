from __future__ import annotations

import unittest

from evals.prepare_sft_dataset import (
    build_sft_record,
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
                    "- Fix 1: Choose one point.",
                    "- Next run: Rehearse once.",
                ]
            ),
        )

    def test_normalize_gold_review_accepts_three_middle_fixes(self) -> None:
        review = "\n".join(
            [
                "• Strength: Specific opening.",
                "• Fix: Cut the generic setup.",
                "• Polish: Slow the toast line.",
                "• Fix: Name the next action.",
                "• Next run: Rehearse once.",
            ]
        )

        self.assertEqual(
            normalize_gold_review(review),
            "\n".join(
                [
                    "- Strength: Specific opening.",
                    "- Fix 1: Cut the generic setup.",
                    "- Fix 2: Slow the toast line.",
                    "- Fix 3: Name the next action.",
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

    def test_build_sft_record_cleans_unfaithful_gold_quotes(self) -> None:
        record = build_sft_record(
            {
                "id": "1",
                "type": "best_man",
                "quality": "test",
                "variant": "quote-cleaning",
                "garble": "none",
                "text": "Sam would jump in first, freezing or not.",
                "gold_review": (
                    '• Strength: "jump in first, freezing or not" is specific.\n'
                    '• Fix: "He froze or not" should be sharper.\n'
                    "• Next run: Rehearse once."
                ),
            },
            {
                "id": "1",
                "computed_word_count": "8",
                "computed_filler_count": "0",
                "computed_notable_fillers": "",
                "computed_filler_counts_json": "{}",
            },
        )

        self.assertEqual(
            record["messages"][2]["content"],
            (
                '- Strength: "jump in first, freezing or not" is specific.\n'
                "- Fix 1: He froze or not should be sharper.\n"
                "- Next run: Rehearse once."
            ),
        )

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

    def test_prepare_records_adds_augments_to_train_and_smoke_only(self) -> None:
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
            augment_rows=[
                {
                    "id": "semantic-canary-001",
                    "type": "best_man",
                    "quality": "semantic_canary",
                    "variant": "actor_negation",
                    "text": "Joe lent me his car for a month and never asked for petrol money.",
                    "gold_review": (
                        "• Strength: Joe lending the car for a month is the proof of generosity.\n"
                        "• Fix: Cut everything before that story.\n"
                        "• Next run: Open with Joe lending you the car."
                    ),
                }
            ],
            seed=42,
            val_fraction=0.25,
            smoke_size=4,
        )

        self.assertEqual(len(records["train"]), 13)
        self.assertEqual(len(records["val"]), 4)
        self.assertIn("semantic-canary-001", {row["source_id"] for row in records["train"]})
        self.assertNotIn("semantic-canary-001", {row["source_id"] for row in records["val"]})
        self.assertIn("semantic-canary-001", {row["source_id"] for row in records["smoke"]})

    def test_prepare_records_repeats_augments_with_unique_train_ids(self) -> None:
        gold_rows = {}
        stats_rows = {}
        for index in range(1, 5):
            row_id = str(index)
            gold_rows[row_id] = {
                "id": row_id,
                "type": "best_man",
                "quality": "weak",
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
            augment_rows=[
                {
                    "id": "perspective-short-001",
                    "type": "best_man",
                    "quality": "perspective_preservation",
                    "augmentation_type": "perspective_preservation_short",
                    "canary": False,
                    "repeat": 2,
                    "text": "Maya lent me her laptop before my interview.",
                    "gold_review": (
                        "• Strength: Maya lending you her laptop is the proof.\n"
                        "• Fix: Keep Maya as the person helping you.\n"
                        "• Next run: Build around Maya lending you her laptop."
                    ),
                }
            ],
            seed=42,
            val_fraction=0.25,
            smoke_size=2,
        )

        repeated = [row for row in records["train"] if row["source_id"].startswith("perspective-short-001")]
        self.assertEqual([row["source_id"] for row in repeated], ["perspective-short-001-repeat-1", "perspective-short-001-repeat-2"])
        self.assertEqual({row["metadata"]["repeat_source_id"] for row in repeated}, {"perspective-short-001"})
        self.assertEqual({row["metadata"]["repeat_count"] for row in repeated}, {2})
        self.assertFalse(any(row["source_id"].startswith("perspective-short-001") for row in records["val"]))

    def test_split_train_val_uses_stratified_groups(self) -> None:
        rows = [
            {"source_id": str(index), "metadata": {"type": "best_man", "quality": "weak"}}
            for index in range(1, 11)
        ]

        train_rows, val_rows = split_train_val(rows, seed=42, val_fraction=0.2)

        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(val_rows), 2)

    def test_split_train_val_keeps_scenario_families_together(self) -> None:
        rows = [
            {
                "source_id": "car-1",
                "metadata": {"type": "best_man", "quality": "weak", "scenario_family": "car-loan"},
            },
            {
                "source_id": "car-2",
                "metadata": {"type": "best_man", "quality": "weak", "scenario_family": "car-loan"},
            },
            *[
                {
                    "source_id": f"other-{index}",
                    "metadata": {"type": "best_man", "quality": "weak"},
                }
                for index in range(1, 9)
            ],
        ]

        train_rows, val_rows = split_train_val(rows, seed=42, val_fraction=0.2)

        train_sources = {row["source_id"] for row in train_rows}
        val_sources = {row["source_id"] for row in val_rows}
        self.assertFalse({"car-1", "car-2"} & train_sources and {"car-1", "car-2"} & val_sources)


if __name__ == "__main__":
    unittest.main()
