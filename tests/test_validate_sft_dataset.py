from __future__ import annotations

import unittest

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
                            "content": "- Strength: Clear.\n- Fix: Add one detail.\n- Next run: Rehearse once.",
                        },
                    ],
                }
            ],
            eval_hashes=set(),
            validate_assistant_scorecard=validate_assistant_scorecard,
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
                            {"role": "assistant", "content": "- Strength: Clear.\n- Fix: Add one detail."},
                        ],
                    }
                ],
                eval_hashes=set(),
                validate_assistant_scorecard=validate_assistant_scorecard,
            )


if __name__ == "__main__":
    unittest.main()
