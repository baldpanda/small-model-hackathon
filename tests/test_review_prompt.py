from __future__ import annotations

import unittest
from unittest.mock import patch

from review import (
    MODEL_ID,
    _build_messages,
    _get_review_adapter_id,
    _model_device,
    _review_model_label,
    clean_review_output,
    format_review_stats,
    is_valid_scorecard_shape,
    scorecard_shape_issues,
)


class ReviewPromptTests(unittest.TestCase):
    def test_review_adapter_id_is_optional(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(_get_review_adapter_id())

        with patch.dict("os.environ", {"REVIEW_ADAPTER_ID": "   "}, clear=True):
            self.assertIsNone(_get_review_adapter_id())

    def test_review_adapter_label_includes_configured_adapter(self) -> None:
        adapter_id = "build-small-hackathon/minicpm5-speech-feedback-lora"

        with patch.dict("os.environ", {"REVIEW_ADAPTER_ID": f" {adapter_id} "}, clear=True):
            self.assertEqual(_get_review_adapter_id(), adapter_id)

        self.assertEqual(_review_model_label(None), MODEL_ID)
        self.assertEqual(_review_model_label(adapter_id), f"{MODEL_ID} + {adapter_id}")

    def test_model_device_uses_parameter_device(self) -> None:
        class FakeParameter:
            device = "cuda:0"

        class FakeModel:
            def parameters(self):
                yield FakeParameter()

        self.assertEqual(_model_device(FakeModel()), "cuda:0")

    def test_format_review_stats_uses_compact_readable_lines(self) -> None:
        stats = {
            "duration_seconds": 95,
            "duration_mmss": "1:35",
            "word_count": 305,
            "wpm": 192,
            "wpm_band": "brisk (181-200)",
            "filler_count": 8,
            "filler_per_min": 5.1,
            "filler_band": "high (5+/min)",
            "notable_fillers": [
                {"filler": "so", "count": 4},
                {"filler": "um", "count": 3},
            ],
        }

        formatted = format_review_stats(stats)

        self.assertIn("- Duration: 1:35 (95.0 seconds)", formatted)
        self.assertIn("- Word count: 305", formatted)
        self.assertIn("- Pace: 192.0 wpm (brisk (181-200))", formatted)
        self.assertIn("- Fillers: 8 total, 5.1 per minute (high (5+/min))", formatted)
        self.assertIn("- Notable fillers: so 4, um 3", formatted)

    def test_build_messages_includes_stats_block_and_transcript(self) -> None:
        messages = _build_messages(
            "To Ava and Sam.",
            {"word_count": 4, "filler_count": 0},
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Stats:", messages[1]["content"])
        self.assertIn("- Word count: 4", messages[1]["content"])
        self.assertIn("Transcript:\nTo Ava and Sam.", messages[1]["content"])
        self.assertIn("Output exactly 4 bullets", messages[0]["content"])
        self.assertIn('"- Strength:", "- Fix 1:", "- Fix 2:", "- Next run:"', messages[0]["content"])
        self.assertIn("Output exactly four hyphen bullets", messages[1]["content"])

    def test_build_messages_instructs_stats_and_functional_role_use(self) -> None:
        messages = _build_messages(
            "Good evening, fellow Toastmasters. Tonight I'm the grammarian.",
            {"wpm": 97.4, "wpm_band": "slow (<120)", "filler_count": 0, "filler_band": "low (0-1/min)"},
        )

        self.assertIn("Toastmasters role such as grammarian", messages[0]["content"])
        self.assertIn("A slow pace under 120 wpm is worth addressing", messages[0]["content"])
        self.assertIn("Do not make both fixes about the same example", messages[0]["content"])
        self.assertIn("Use Fix 2 for delivery or stats", messages[1]["content"])
        self.assertIn("- Pace: 97.4 wpm (slow (<120))", messages[1]["content"])

    def test_clean_review_output_stops_after_first_next_run(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The opening image is specific.",
                    "- Fix 1: Move the main story earlier.",
                    "- Pace: 205 wpm is fast.",
                    "- Fix 2: Cut the weakest aside.",
                    "- Next run: Rehearse the story into the toast.",
                    "- Fix 1: Extra analysis that should be dropped.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The opening image is specific.",
                    "- Fix 1: Move the main story earlier.",
                    "- Fix 2: Cut the weakest aside.",
                    "- Next run: Rehearse the story into the toast.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_clean_review_output_normalizes_old_middle_labels(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix: Make the preference explicit.",
                    "- Polish: Slow the opening from 205 wpm.",
                    "- Next run: Say the final sentence once without fillers.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the preference explicit.",
                    "- Fix 2: Slow the opening from 205 wpm.",
                    "- Next run: Say the final sentence once without fillers.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_clean_review_output_normalizes_nested_next_run_label(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix: Fix 1: Make the preference explicit.",
                    "- Fix: Fix 2: Reduce the filler-heavy opening.",
                    "- Fix: Next run: Say the final sentence once without fillers.",
                    "- Fix: Extra advice should be dropped.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the preference explicit.",
                    "- Fix 2: Reduce the filler-heavy opening.",
                    "- Next run: Say the final sentence once without fillers.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_scorecard_shape_issues_require_final_next_run(self) -> None:
        issues = scorecard_shape_issues(
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the preference explicit.",
                    "- Fix 2: Slow the opening.",
                ]
            )
        )

        self.assertIn("expected 4 lines, found 3", issues)
        self.assertFalse(is_valid_scorecard_shape("\n".join(["- Strength: Clear.", "- Fix 1: Specific."])))


if __name__ == "__main__":
    unittest.main()
