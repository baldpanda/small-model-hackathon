from __future__ import annotations

import unittest

from evals.validate_sft_dataset import assistant_fix_count_distribution
from evals.prepare_sft_dataset import validate_assistant_scorecard
from evals.validate_sft_dataset import validate_rows


class ValidateSftDatasetTests(unittest.TestCase):
    def test_validate_rows_accepts_valid_messages(self) -> None:
        validate_rows(
            [
                {
                    "id": "train-001",
                    "source_id": "1",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "Stats:\n- Word count: 4\n\nTranscript:\nHello there."},
                        {
                            "role": "assistant",
                            "content": "- Strength: Clear.\n- Fix 1: Add one detail.\n- Next run: Rehearse once.",
                        },
                    ],
                }
            ],
            eval_hashes=set(),
            validate_assistant_scorecard=validate_assistant_scorecard,
        )

    def test_validate_rows_accepts_exact_quotes_when_quote_check_enabled(self) -> None:
        validate_rows(
            [
                {
                    "id": "train-001",
                    "source_id": "1",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": "Stats:\n- Word count: 8\n\nTranscript:\nSam would jump in first, freezing or not.",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                '- Strength: "jump in first, freezing or not" is specific.\n'
                                "- Fix 1: Add one detail.\n"
                                "- Next run: Rehearse once."
                            ),
                        },
                    ],
                }
            ],
            eval_hashes=set(),
            validate_assistant_scorecard=validate_assistant_scorecard,
            check_quote_faithfulness=True,
        )

    def test_validate_rows_rejects_misquotes_when_quote_check_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "quote-faithfulness failed"):
            validate_rows(
                [
                    {
                        "id": "train-001",
                        "source_id": "1",
                        "messages": [
                            {"role": "system", "content": "system"},
                            {
                                "role": "user",
                                "content": "Stats:\n- Word count: 8\n\nTranscript:\nSam would jump in first, freezing or not.",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    '- Strength: "He froze or not" is specific.\n'
                                    "- Fix 1: Add one detail.\n"
                                    "- Next run: Rehearse once."
                                ),
                            },
                        ],
                    }
                ],
                eval_hashes=set(),
                validate_assistant_scorecard=validate_assistant_scorecard,
                check_quote_faithfulness=True,
            )

    def test_validate_rows_rejects_missing_next_run(self) -> None:
        with self.assertRaises(ValueError):
            validate_rows(
                [
                    {
                        "id": "train-001",
                        "source_id": "1",
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "Stats:\n- Word count: 4\n\nTranscript:\nHello there."},
                            {"role": "assistant", "content": "- Strength: Clear.\n- Fix 1: Add one detail."},
                        ],
                    }
                ],
                eval_hashes=set(),
                validate_assistant_scorecard=validate_assistant_scorecard,
            )

    def test_assistant_fix_count_distribution_counts_middle_bullets(self) -> None:
        counts = assistant_fix_count_distribution(
            [
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "user"},
                        {
                            "role": "assistant",
                            "content": "- Strength: Clear.\n- Fix 1: Add detail.\n- Next run: Rehearse.",
                        },
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "user"},
                        {
                            "role": "assistant",
                            "content": (
                                "- Strength: Clear.\n"
                                "- Fix 1: Add detail.\n"
                                "- Fix 2: Slow down.\n"
                                "- Next run: Rehearse."
                            ),
                        },
                    ]
                },
            ]
        )

        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 1)


if __name__ == "__main__":
    unittest.main()
