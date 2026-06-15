from __future__ import annotations

import unittest

from evals.check_review_expectations import check_expectations


class CheckReviewExpectationsTests(unittest.TestCase):
    def test_check_expectations_accepts_matching_review(self) -> None:
        records = [
            {
                "id": "row-1",
                "expectations": {
                    "must_include_any": [["Maya lent you her laptop", "Maya lending you her laptop"]],
                    "must_not_include": ["you lent Maya your laptop"],
                },
            }
        ]
        reviews = {
            "row-1": {
                "review": "- Strength: Maya lending you her laptop is the proof.\n- Fix 1: Keep it.\n- Next run: Say it.",
                "scorecard_shape_valid": True,
                "quote_faithfulness_valid": True,
            }
        }

        self.assertEqual(check_expectations(records, reviews), [])

    def test_check_expectations_flags_forbidden_perspective_flip(self) -> None:
        records = [
            {
                "id": "row-1",
                "expectations": {
                    "must_include_any": [["Maya lent you her laptop"]],
                    "must_not_include": ["you lent Maya your laptop"],
                },
            }
        ]
        reviews = {
            "row-1": {
                "review": "- Strength: You lent Maya your laptop.",
                "scorecard_shape_valid": True,
                "quote_faithfulness_valid": True,
            }
        }

        failures = check_expectations(records, reviews)

        self.assertIn("row-1: missing one of ['Maya lent you her laptop']", failures)
        self.assertIn("row-1: contains forbidden phrase 'you lent Maya your laptop'", failures)


if __name__ == "__main__":
    unittest.main()
