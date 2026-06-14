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
    _review_generate_kwargs,
    _render_chat_prompt_for_logging,
    _review_model_label,
    _should_log_review_prompt,
    clean_unfaithful_review_quotes,
    clean_review_output,
    format_review_stats,
    is_valid_scorecard_shape,
    quote_faithfulness_issues,
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
        self.assertIn("Output exactly 3-5 bullets", messages[0]["content"])
        self.assertIn('"- Strength:", then "- Fix 1:"', messages[0]["content"])
        self.assertIn("optional \"- Fix 2:\" and \"- Fix 3:\"", messages[0]["content"])
        self.assertIn("Output exactly 3-5 hyphen bullets", messages[1]["content"])

    def test_build_messages_prefers_one_fix_by_default(self) -> None:
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

        self.assertIn("Use one fix by default", messages[0]["content"])
        self.assertIn("Mention pace, duration, fillers, or other stats only when", messages[0]["content"])
        self.assertIn("Use one fix by default", messages[1]["content"])
        self.assertIn("Mention stats only when", messages[1]["content"])

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
        self.assertIn("Do not make multiple fixes about the same example", messages[0]["content"])
        self.assertIn("Do not repeat feedback in different words", messages[0]["content"])
        self.assertIn("Add more fixes only when they are distinct", messages[1]["content"])
        self.assertIn("Do not repeat feedback", messages[1]["content"])
        self.assertIn("- Pace: 97.4 wpm (slow (<120))", messages[1]["content"])

    def test_build_messages_treats_very_fast_pace_as_high_impact(self) -> None:
        messages = _build_messages(
            "Good evening, fellow Toastmasters and guests.",
            {
                "word_count": 87,
                "wpm": 200.9,
                "wpm_band": "fast (>200)",
                "filler_count": 0,
                "filler_band": "low (0-1/min)",
            },
        )

        self.assertIn("Fast pace above 200 wpm", messages[0]["content"])
        self.assertIn("fast pace above 200 wpm", messages[1]["content"])
        self.assertIn("- Pace: 200.9 wpm (fast (>200))", messages[1]["content"])
        self.assertIn("Use double quotes only for exact spans from the transcript", messages[0]["content"])
        self.assertIn("Only use double quotes for exact transcript spans", messages[1]["content"])

    def test_review_generate_kwargs_include_repetition_guardrails(self) -> None:
        class FakeTokenizer:
            eos_token_id = 123

        generate_kwargs = _review_generate_kwargs(FakeTokenizer())

        self.assertEqual(generate_kwargs["repetition_penalty"], 1.2)
        self.assertEqual(generate_kwargs["no_repeat_ngram_size"], 0)
        self.assertEqual(generate_kwargs["pad_token_id"], 123)

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

    def test_clean_review_output_dedupes_repeated_fixes(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The lake image is specific.",
                    "- Fix 1: The generic generosity line needs one concrete example.",
                    "- Fix 2: The generic generosity line needs one concrete example.",
                    "- Fix 3: Slow the 214 wpm pace so the joke has room.",
                    "- Next run: Replace the generosity line and rehearse at 170 wpm.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The lake image is specific.",
                    "- Fix 1: The generic generosity line needs one concrete example.",
                    "- Fix 2: Slow the 214 wpm pace so the joke has room.",
                    "- Next run: Replace the generosity line and rehearse at 170 wpm.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_clean_review_output_drops_fix_that_repeats_strength(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The compost image is vivid and specific.",
                    "- Fix 1: The compost image is vivid and specific.",
                    "- Fix 2: Add one sentence explaining why the garden matters.",
                    "- Next run: Rehearse the compost beat once without adding examples.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The compost image is vivid and specific.",
                    "- Fix 1: Add one sentence explaining why the garden matters.",
                    "- Next run: Rehearse the compost beat once without adding examples.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_clean_review_output_keeps_one_fix_when_model_outputs_one(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the point specific.",
                    "- Next run: Record one complete sentence.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The topic is clear.",
                    "- Fix 1: Make the point specific.",
                    "- Next run: Record one complete sentence.",
                ]
            ),
        )
        self.assertTrue(is_valid_scorecard_shape(cleaned))

    def test_clean_review_output_strips_thinking_and_normalizes_plain_labels(self) -> None:
        cleaned = clean_review_output(
            "\n".join(
                [
                    "<think>",
                    "",
                    "</think> Strength: The role is clear.",
                    "Fix 1: Add one concrete detail.",
                    "Fix 2: Add another example.",
                    "Fix 3: Cut the repeated line.",
                    "Next run: Record one complete sentence.",
                ]
            )
        )

        self.assertEqual(
            cleaned,
            "\n".join(
                [
                    "- Strength: The role is clear.",
                    "- Fix 1: Add one concrete detail.",
                    "- Fix 2: Add another example.",
                    "- Fix 3: Cut the repeated line.",
                    "- Next run: Record one complete sentence.",
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
                    "- Fix: Fix 3: Cut the repeated ending.",
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
                    "- Fix 3: Cut the repeated ending.",
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
                ]
            )
        )

        self.assertIn("expected 3-5 lines, found 2", issues)
        self.assertFalse(is_valid_scorecard_shape("\n".join(["- Strength: Clear.", "- Fix 1: Specific."])))

    def test_scorecard_shape_issues_accepts_one_fix_contract(self) -> None:
        review = "\n".join(
            [
                "- Strength: The role is clear.",
                "- Fix 1: Add one concrete detail.",
                "- Next run: Record the opening once.",
            ]
        )

        self.assertEqual(scorecard_shape_issues(review), [])
        self.assertTrue(is_valid_scorecard_shape(review))

    def test_quote_faithfulness_issues_accept_exact_transcript_quote(self) -> None:
        transcript = "Sam was always the one who'd jump in first, freezing or not."
        review = '- Strength: "jump in first, freezing or not" is the real image.'

        self.assertEqual(quote_faithfulness_issues(review, transcript), [])

    def test_quote_faithfulness_issues_flags_misquote(self) -> None:
        transcript = "Sam was always the one who'd jump in first, freezing or not."
        review = '- Strength: "He froze or not" is the real image.'

        self.assertEqual(
            quote_faithfulness_issues(review, transcript),
            ['quoted span not found in transcript: "He froze or not"'],
        )

    def test_quote_faithfulness_issues_ignores_short_labels(self) -> None:
        transcript = "This has no filler words."
        review = '- Fix 1: Replace "um" only if it appears.'

        self.assertEqual(quote_faithfulness_issues(review, transcript), [])

    def test_clean_unfaithful_review_quotes_removes_only_unmatched_quote_marks(self) -> None:
        transcript = "Sam would jump in first, freezing or not."
        review = (
            '- Strength: "jump in first, freezing or not" is specific.\n'
            '- Fix 1: "He froze or not" should be sharper.'
        )

        self.assertEqual(
            clean_unfaithful_review_quotes(review, transcript),
            (
                '- Strength: "jump in first, freezing or not" is specific.\n'
                "- Fix 1: He froze or not should be sharper."
            ),
        )


if __name__ == "__main__":
    unittest.main()
