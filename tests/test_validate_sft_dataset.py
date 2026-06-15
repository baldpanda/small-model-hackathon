from __future__ import annotations

import unittest

from evals.validate_sft_dataset import assistant_fix_count_distribution
from evals.validate_sft_dataset import fix_distinctness_issues
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

    def test_fix_distinctness_accepts_distinct_fixes(self) -> None:
        assistant = (
            "- Strength: Clear.\n"
            "- Fix 1: Open with the car story so the proof arrives first.\n"
            "- Fix 2: Slow the toast line because the pace is brisk.\n"
            "- Next run: Rehearse once."
        )

        self.assertEqual(fix_distinctness_issues(assistant), [])

    def test_fix_distinctness_flags_near_duplicate_fixes(self) -> None:
        assistant = (
            "- Strength: Clear.\n"
            "- Fix 1: Add one concrete story about Joe lending you the car.\n"
            "- Fix 2: Add one specific story about Joe lending you the car.\n"
            "- Next run: Rehearse once."
        )

        issues = fix_distinctness_issues(assistant, max_similarity=0.78)

        self.assertEqual(len(issues), 1)
        self.assertIn("too similar", issues[0])

    def test_validate_rows_rejects_duplicate_fix_gold_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "fix-distinctness failed"):
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
                                "content": (
                                    "- Strength: Clear.\n"
                                    "- Fix 1: Add one concrete story.\n"
                                    "- Fix 2: Add one concrete story.\n"
                                    "- Next run: Rehearse once."
                                ),
                            },
                        ],
                    }
                ],
                eval_hashes=set(),
                validate_assistant_scorecard=validate_assistant_scorecard,
                check_distinct_fixes=True,
            )


if __name__ == "__main__":
    unittest.main()
