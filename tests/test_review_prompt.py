from __future__ import annotations

import unittest
from unittest.mock import patch

from review import (
    MODEL_ID,
    _build_messages,
    _format_messages_for_logging,
    _get_review_adapter_id,
    _log_review_prompt,
    _model_device,
    _render_chat_prompt_for_logging,
    _review_model_label,
    _should_log_review_prompt,
    clean_review_output,
    expected_scorecard_labels,
    format_review_stats,
    is_valid_scorecard_shape,
    scorecard_shape_issues,
)


class ReviewPromptTests(unittest.TestCase):
    def test_review_prompt_logging_is_opt_in(self) -> None:
        truthy_values = ["1", "true", "yes", "on"]
        for value in truthy_values:
            with self.subTest(value=value):
                with patch.dict("os.environ", {"REVIEW_LOG_PROMPT": value}, clear=True):
                    self.assertTrue(_should_log_review_prompt())

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_should_log_review_prompt())

        with patch.dict("os.environ", {"REVIEW_LOG_PROMPT": "0"}, clear=True):
            self.assertFalse(_should_log_review_prompt())

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

    def test_render_chat_prompt_for_logging_uses_non_tokenized_chat_template(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.kwargs = {}

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered prompt"

        tokenizer = FakeTokenizer()
        rendered = _render_chat_prompt_for_logging(tokenizer, [{"role": "user", "content": "hello"}])

        self.assertEqual(rendered, "rendered prompt")
        self.assertFalse(tokenizer.kwargs["tokenize"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_prompt_logging_falls_back_to_messages(self) -> None:
        class BrokenTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                raise RuntimeError("template failed")

        messages = [{"role": "user", "content": "hello"}]

        with self.assertLogs("review", level="WARNING") as logs:
            rendered = _render_chat_prompt_for_logging(BrokenTokenizer(), messages)

        self.assertEqual(rendered, "USER:\nhello")
        self.assertIn("Failed to render chat template", "\n".join(logs.output))

    def test_log_review_prompt_respects_env_flag(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "rendered prompt"

        messages = [{"role": "user", "content": "hello"}]

        with patch.dict("os.environ", {}, clear=True):
            with self.assertNoLogs("review", level="WARNING"):
                _log_review_prompt(FakeTokenizer(), messages)

        with patch.dict("os.environ", {"REVIEW_LOG_PROMPT": "1"}, clear=True):
            with self.assertLogs("review", level="WARNING") as logs:
                _log_review_prompt(FakeTokenizer(), messages)

        self.assertIn("REVIEW_LOG_PROMPT is enabled", "\n".join(logs.output))
        self.assertIn("rendered prompt", "\n".join(logs.output))

    def test_format_messages_for_logging_uses_roles(self) -> None:
        self.assertEqual(
            _format_messages_for_logging(
                [
                    {"role": "system", "content": "coach"},
                    {"role": "user", "content": "speech"},
                ]
            ),
            "SYSTEM:\ncoach\n\nUSER:\nspeech",
        )

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
            {"word_count": 80, "filler_count": 0},
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Stats:", messages[1]["content"])
        self.assertIn("- Word count: 80", messages[1]["content"])
        self.assertIn("Transcript:\nTo Ava and Sam.", messages[1]["content"])
        self.assertIn("Output exactly 4 bullets", messages[0]["content"])
        self.assertIn('"- Strength:", "- Fix 1:", "- Fix 2:", "- Next run:"', messages[0]["content"])
        self.assertIn("Output exactly four hyphen bullets", messages[1]["content"])

    def test_build_messages_uses_short_contract_for_tiny_clip(self) -> None:
        messages = _build_messages(
            "Into the back row, and what about Lewis Hamilton in that row?",
            {
                "duration_seconds": 13.3,
                "word_count": 28,
                "wpm": 126.7,
                "wpm_band": "medium (120-180)",
                "filler_count": 0,
            },
        )

        self.assertEqual(expected_scorecard_labels("short clip", {"word_count": 28}), ("Strength", "Fix", "Next run"))
        self.assertIn("Output exactly 3 bullets", messages[0]["content"])
        self.assertIn('"- Strength:", "- Fix:", "- Next run:"', messages[0]["content"])
        self.assertIn("Do not output \"Fix 1:\" or \"Fix 2:\"", messages[0]["content"])
        self.assertIn("Output exactly three hyphen bullets", messages[1]["content"])
        self.assertIn("Use Fix for the single highest-impact change", messages[1]["content"])

    def test_build_messages_instructs_stats_and_functional_role_use(self) -> None:
        messages = _build_messages(
            "Good evening, fellow Toastmasters. Tonight I'm the grammarian.",
            {
                "word_count": 90,
                "wpm": 97.4,
                "wpm_band": "slow (<120)",
                "filler_count": 0,
                "filler_band": "low (0-1/min)",
            },
        )

        self.assertIn("Toastmasters role such as grammarian", messages[0]["content"])
        self.assertIn("A slow pace under 120 wpm is worth addressing", messages[0]["content"])
        self.assertIn("Do not make both fixes about the same example", messages[0]["content"])
        self.assertIn("Do not repeat feedback in different words", messages[0]["content"])
        self.assertIn("Use Fix 2 for delivery or stats", messages[1]["content"])
        self.assertIn("Do not repeat feedback", messages[1]["content"])
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

    def test_clean_review_output_uses_one_fix_for_short_contract(self) -> None:
        expected_labels = ("Strength", "Fix", "Next run")
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the point specific.",
                    "- Fix 2: Add another example.",
                    "- Next run: Record one complete sentence.",
                ]
            ),
            expected_labels=expected_labels,
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix: Make the point specific.",
                    "- Next run: Record one complete sentence.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned, expected_labels=expected_labels))

    def test_clean_review_output_strips_thinking_and_normalizes_plain_short_labels(self) -> None:
        expected_labels = ("Strength", "Fix", "Next run")
        cleaned = clean_review_output(
            "\n".join(
                [
                    "<think>",
                    "",
                    "</think> Strength: The role is clear.",
                    "Fix 1: Add one concrete detail.",
                    "Fix 2: Add another example.",
                    "Next run: Record one complete sentence.",
                ]
            ),
            expected_labels=expected_labels,
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The role is clear.",
                    "- Fix: Add one concrete detail.",
                    "- Next run: Record one complete sentence.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned, expected_labels=expected_labels))

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

    def test_scorecard_shape_issues_accept_short_contract_when_expected(self) -> None:
        review = "\n".join(
            [
                "- Strength: The role is clear.",
                "- Fix: Add one concrete detail.",
                "- Next run: Record the opening once.",
            ]
        )
        expected_labels = ("Strength", "Fix", "Next run")

        self.assertEqual(scorecard_shape_issues(review, expected_labels=expected_labels), [])
        self.assertTrue(is_valid_scorecard_shape(review, expected_labels=expected_labels))


if __name__ == "__main__":
    unittest.main()
