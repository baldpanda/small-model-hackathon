from __future__ import annotations

import unittest

from rehearsal_limits import (
    SHORT_RECORDING_MESSAGE,
    estimate_gpu_duration_seconds,
    validate_recording_duration_seconds,
)


class RehearsalLimitsTests(unittest.TestCase):
    def test_rejects_recordings_under_minimum_with_encouraging_message(self) -> None:
        with self.assertRaisesRegex(ValueError, SHORT_RECORDING_MESSAGE):
            validate_recording_duration_seconds(9.9)

    def test_accepts_recordings_at_window_edges(self) -> None:
        validate_recording_duration_seconds(10)
        validate_recording_duration_seconds(120)

    def test_rejects_recordings_over_maximum_with_two_minute_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "longer than 2 minutes"):
            validate_recording_duration_seconds(120.1)

    def test_estimates_gpu_duration_with_pilot_clamps(self) -> None:
        self.assertEqual(estimate_gpu_duration_seconds(10), 15)
        self.assertEqual(estimate_gpu_duration_seconds(45), 19)
        self.assertEqual(estimate_gpu_duration_seconds(120), 30)


if __name__ == "__main__":
    unittest.main()
