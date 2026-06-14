import logging
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import spaces

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from filler_words import summarize_fillers
from rehearsal_limits import (
    MAX_RECORDING_SECONDS,
    accepted_recording_window_label,
    estimate_gpu_duration_seconds,
    validate_recording_duration_seconds,
)
from review import review_speech
from speech_stats import build_transcript_stats
from timing import get_audio_duration_seconds, summarize_timing
from transcribe import transcribe_recording


APP_DIR = Path(__file__).parent
RECORDING_WINDOW_LABEL = accepted_recording_window_label()
SPEECH_FEEDBACK_PENDING = "_Speech feedback will appear after the model review finishes._"
TIMING_FEEDBACK_PENDING = "_Timing feedback will appear after transcription._"
FILLER_FEEDBACK_PENDING = "_Filler feedback will appear after transcription._"


def _format_clock_seconds(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"

COUNTDOWN_HEAD = f"""
<script>
(() => {{
  const limitSeconds = {MAX_RECORDING_SECONDS};
  let timerId = null;
  let secondsLeft = limitSeconds;
  let lastButton = null;

  const formatSeconds = (value) => {{
    const minutes = Math.floor(value / 60);
    const seconds = value % 60;
    return `${{minutes}}:${{String(seconds).padStart(2, "0")}}`;
  }};

  const updateCountdown = (message) => {{
    const status = document.querySelector("#recording-status");
    if (status) {{
      status.textContent = message;
    }}
  }};

  const stopTimer = (message) => {{
    if (timerId) {{
      window.clearInterval(timerId);
      timerId = null;
    }}
    secondsLeft = limitSeconds;
    lastButton = null;
    updateCountdown(message ?? `Recording limit: ${{formatSeconds(limitSeconds)}}`);
  }};

  const stopRecording = () => {{
    if (lastButton) {{
      lastButton.click();
    }}
  }};

  const startTimer = (button) => {{
    if (timerId) {{
      return;
    }}

    lastButton = button;
    secondsLeft = limitSeconds;
    updateCountdown(`Recording... ${{formatSeconds(secondsLeft)}} remaining`);

    timerId = window.setInterval(() => {{
      secondsLeft -= 1;
      if (secondsLeft <= 0) {{
        stopTimer(`Reached ${{formatSeconds(limitSeconds)}}. Recording stopped. Review when ready.`);
        stopRecording();
        return;
      }}

      updateCountdown(`Recording... ${{formatSeconds(secondsLeft)}} remaining`);
    }}, 1000);
  }};

  const wireAudioButtons = () => {{
    const root = document.querySelector("#speech-audio");
    if (!root) {{
      window.setTimeout(wireAudioButtons, 500);
      return;
    }}

    root.addEventListener("click", (event) => {{
      const button = event.target.closest("button");
      if (!button) {{
        return;
      }}

      window.setTimeout(() => {{
        const label = (button.getAttribute("aria-label") || button.innerText || "").toLowerCase();
        if (label.includes("record") || label.includes("stop")) {{
          if (timerId) {{
            stopTimer("Recording stopped. Review when ready.");
          }} else {{
            startTimer(button);
          }}
        }}
      }}, 0);
    }});
  }};

  window.addEventListener("load", wireAudioButtons);
  window.addEventListener("beforeunload", () => stopTimer());
}})();
</script>
"""

CUSTOM_CSS = (APP_DIR / "assets" / "scorecard.css").read_text()

SPEECH_FEEDBACK_SECTIONS = (
    "What worked",
    "What to sharpen",
    "Try this next time",
    "Bottom line",
)


class ProcessingTimer:
    def __init__(self, requested_gpu_seconds: int | None = None) -> None:
        self.started_at = time.perf_counter()
        self.requested_gpu_seconds = requested_gpu_seconds
        self.steps: list[tuple[str, float]] = []

    def add_step(self, label: str, seconds: float) -> None:
        self.steps.append((label, seconds))

    def total_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def format_markdown(self) -> str:
        lines = ["", "**Processing timings**"]
        if self.requested_gpu_seconds is not None:
            lines.append(f"- requested GPU budget: {self.requested_gpu_seconds}s")
        for label, seconds in self.steps:
            lines.append(f"- {label}: {seconds:.1f}s")
        lines.append(f"- total: {self.total_seconds():.1f}s")
        return "\n".join(lines)

    def log(self, status: str) -> None:
        parts = [f"{label}={seconds:.1f}s" for label, seconds in self.steps]
        if self.requested_gpu_seconds is not None:
            parts.append(f"requested_gpu_budget={self.requested_gpu_seconds}s")
        parts.append(f"total={self.total_seconds():.1f}s")
        LOGGER.info("process_rehearsal status=%s timings=%s", status, " ".join(parts))


def _gpu_duration_seconds(audio_path: str, duration_seconds: float, requested_gpu_seconds: int) -> int:
    return requested_gpu_seconds


def _timed_step(timer: ProcessingTimer, label: str, action):
    step_started_at = time.perf_counter()
    try:
        return action()
    finally:
        timer.add_step(label, time.perf_counter() - step_started_at)


def _status_with_timings(message: str, timer: ProcessingTimer, *, log: bool = True) -> str:
    if log:
        timer.log(message.splitlines()[0])
    return f"{message}{timer.format_markdown()}"


def _outputs(
    transcript: str,
    feedback: str,
    timing_feedback: str,
    filler_feedback: str,
    status: str,
    timer: ProcessingTimer,
    *,
    log_status: bool = False,
) -> tuple[str, str, str, str, str]:
    return (
        transcript,
        feedback,
        timing_feedback,
        filler_feedback,
        _status_with_timings(status, timer, log=log_status),
    )


def _format_final_outputs(
    transcript: str,
    feedback: str,
    timing_feedback: str,
    filler_feedback: str,
    status: str,
    timer: ProcessingTimer,
) -> tuple[str, str, str, str, str]:
    step_started_at = time.perf_counter()
    formatted_feedback = _format_speech_feedback_markdown(feedback)
    formatted_timing = _format_metric_markdown(timing_feedback)
    formatted_filler = _format_metric_markdown(filler_feedback)
    timer.add_step("formatting", time.perf_counter() - step_started_at)

    return (
        transcript,
        formatted_feedback,
        formatted_timing,
        formatted_filler,
        _status_with_timings(status, timer),
    )


def _format_duration_preview(duration_seconds: float) -> str:
    return _format_metric_markdown(
        "\n".join(
            (
                f"Duration: {duration_seconds:.1f} seconds",
                "Estimated words: pending until transcription completes",
                "Estimated pace: pending until transcription completes",
            )
        )
    )


@spaces.GPU(duration=_gpu_duration_seconds)
def _process_valid_rehearsal(
    audio_path: str,
    duration_seconds: float,
    requested_gpu_seconds: int,
) -> Iterator[tuple[str, str, str, str, str]]:
    timer = ProcessingTimer(requested_gpu_seconds=requested_gpu_seconds)
    timing_preview = _format_duration_preview(duration_seconds)
    yield _outputs(
        "",
        SPEECH_FEEDBACK_PENDING,
        timing_preview,
        FILLER_FEEDBACK_PENDING,
        (
            f"Recording received. Duration: {duration_seconds:.1f}s. "
            f"Requested GPU budget: {requested_gpu_seconds}s. Starting transcription."
        ),
        timer,
    )

    try:
        transcript = _timed_step(
            timer,
            "transcription",
            lambda: transcribe_recording(audio_path, duration_seconds=duration_seconds),
        )
    except ValueError as exc:
        yield "", "", timing_preview, "", _status_with_timings(str(exc), timer)
        return
    except RuntimeError as exc:
        yield "", "", timing_preview, "", _status_with_timings(str(exc), timer)
        return
    except Exception as exc:
        yield "", "", timing_preview, "", _status_with_timings(f"Transcription failed: {exc}", timer)
        return

    yield _outputs(
        transcript,
        SPEECH_FEEDBACK_PENDING,
        timing_preview,
        FILLER_FEEDBACK_PENDING,
        "Transcript ready. Calculating timing feedback.",
        timer,
    )
    timing_feedback = _timed_step(timer, "timing analysis", lambda: _build_timing_feedback(audio_path, transcript))
    formatted_timing_feedback = _format_metric_markdown(timing_feedback)

    yield _outputs(
        transcript,
        SPEECH_FEEDBACK_PENDING,
        formatted_timing_feedback,
        FILLER_FEEDBACK_PENDING,
        "Timing feedback ready. Counting filler words.",
        timer,
    )
    filler_feedback = _timed_step(timer, "filler analysis", lambda: _build_filler_feedback(transcript))
    formatted_filler_feedback = _format_metric_markdown(filler_feedback)

    yield _outputs(
        transcript,
        SPEECH_FEEDBACK_PENDING,
        formatted_timing_feedback,
        formatted_filler_feedback,
        "Filler feedback ready. Generating speech feedback.",
        timer,
    )

    try:
        review_stats = build_transcript_stats(transcript, duration_seconds=duration_seconds)
        feedback = _timed_step(
            timer,
            "review generation",
            lambda: review_speech(transcript, stats=review_stats),
        )
    except ValueError as exc:
        yield _format_final_outputs(
            transcript,
            str(exc),
            timing_feedback,
            filler_feedback,
            "Transcription complete. Review failed.",
            timer,
        )
        return
    except RuntimeError as exc:
        yield _format_final_outputs(
            transcript,
            str(exc),
            timing_feedback,
            filler_feedback,
            "Transcription complete. Review failed.",
            timer,
        )
        return
    except Exception as exc:
        yield _format_final_outputs(
            transcript,
            f"Review failed: {exc}",
            timing_feedback,
            filler_feedback,
            "Transcription complete. Review failed.",
            timer,
        )
        return

    yield _format_final_outputs(
        transcript,
        feedback,
        timing_feedback,
        filler_feedback,
        "Transcription, review, timing, and filler analysis complete.",
        timer,
    )


def _clear_outputs(status: str) -> tuple[str, str, str, str, str]:
    return "", "", "", "", status


def _reset_rehearsal() -> tuple[None, str, str, str, str, str]:
    return (None, *_clear_outputs("Ready to record."))


def _review_button_state(audio_path: str | None):
    return gr.update(interactive=bool(audio_path))


def process_rehearsal(audio_path: str | None) -> Iterator[tuple[str, str, str, str, str]]:
    if not audio_path:
        yield _clear_outputs(f"Record a speech first. The app accepts recordings from {RECORDING_WINDOW_LABEL}.")
        return

    try:
        duration_seconds = get_audio_duration_seconds(audio_path)
        validate_recording_duration_seconds(duration_seconds)
    except ValueError as exc:
        yield _clear_outputs(str(exc))
        return
    except Exception as exc:
        yield _clear_outputs(f"Could not read the recording duration, so no GPU work was requested: {exc}")
        return

    requested_gpu_seconds = estimate_gpu_duration_seconds(duration_seconds)
    yield from _process_valid_rehearsal(audio_path, duration_seconds, requested_gpu_seconds)


def _build_timing_feedback(audio_path: str, transcript: str) -> str:
    try:
        return summarize_timing(audio_path, transcript)
    except Exception as exc:
        return f"Timing analysis failed: {exc}"


def _build_filler_feedback(transcript: str) -> str:
    try:
        return summarize_fillers(transcript)
    except Exception as exc:
        return f"Filler analysis failed: {exc}"


def _format_speech_feedback_markdown(feedback: str) -> str:
    lines: list[str] = []
    for line in feedback.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in SPEECH_FEEDBACK_SECTIONS:
            if lines:
                lines.append("")
            lines.append(f"**{stripped}**")
            lines.append("")
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _format_metric_markdown(summary: str) -> str:
    lines: list[str] = []
    for line in summary.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            lines.append(stripped)
        elif ":" in stripped:
            label, value = stripped.split(":", 1)
            lines.append(f"- **{label}:**{value}")
        else:
            lines.append(stripped)
    return "\n".join(lines)


with gr.Blocks(title="Best Man Speech Coach", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="wedding-app"):
        gr.HTML(
            f"""
            <section id="hero-panel">
              <div class="kicker">Build Small Hackathon rehearsal desk</div>
              <h1>Best Man Speech Coach</h1>
              <p>
                Record a two-minute run-through and get a wedding-scorecard readout:
                transcript, structure notes, pacing, and filler habits before the next toast.
              </p>
              <div class="stamp">Off-brand Gradio edition</div>
            </section>
            """
        )

        with gr.Row(elem_id="practice-grid"):
            with gr.Column(scale=7, elem_classes=["scorecard-card"]):
                gr.Markdown(
                    "## Rehearsal Booth\n"
                    f"Speak naturally. The app accepts recordings from {RECORDING_WINDOW_LABEL}."
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speech recording",
                    elem_id="speech-audio",
                    editable=False,
                    buttons=[],
                )
                countdown = gr.HTML(
                    f"<div id='recording-status'>Recording limit: {_format_clock_seconds(MAX_RECORDING_SECONDS)}</div>",
                    label="Recording timer",
                )
                with gr.Row(elem_id="rehearsal-actions"):
                    transcribe_button = gr.Button(
                        "Review speech",
                        variant="primary",
                        elem_id="review-button",
                        scale=3,
                        interactive=False,
                    )
                    try_again_button = gr.Button(
                        "Try again",
                        variant="secondary",
                        elem_id="try-again-button",
                        scale=1,
                    )

            with gr.Column(scale=5, elem_classes=["scorecard-card"]):
                gr.Markdown("## Scorecard Status")
                status_output = gr.Markdown(
                    value="Ready to record.",
                    elem_id="status-output",
                )

        with gr.Row(elem_id="results-grid"):
            with gr.Column(scale=6, elem_classes=["scorecard-card", "result-panel"]):
                gr.Markdown("## Transcript")
                transcript_output = gr.Textbox(
                    label="Rehearsal transcript",
                    lines=14,
                    placeholder="Your transcript will appear here after recording.",
                    elem_id="transcript-output",
                )

            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Timing Feedback")
                timing_output = gr.Markdown(
                    value="_Timing feedback will appear here after transcription._",
                    elem_id="timing-output",
                    elem_classes=["score-output"],
                )

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Filler Feedback")
                filler_output = gr.Markdown(
                    value="_Filler feedback will appear here after transcription._",
                    elem_classes=["score-output"],
                )

            with gr.Column(scale=6, elem_classes=["scorecard-card", "result-panel"]):
                gr.Markdown("## Speech Feedback")
                feedback_output = gr.Markdown(
                    value="_Speech feedback will appear here after transcription._",
                    elem_classes=["score-output"],
                )

        transcribe_button.click(
            fn=process_rehearsal,
            inputs=audio_input,
            outputs=[transcript_output, feedback_output, timing_output, filler_output, status_output],
        )

        audio_input.change(
            fn=_review_button_state,
            inputs=audio_input,
            outputs=transcribe_button,
        )

        audio_input.start_recording(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=transcribe_button,
        )

        try_again_button.click(
            fn=_reset_rehearsal,
            inputs=None,
            outputs=[
                audio_input,
                transcript_output,
                feedback_output,
                timing_output,
                filler_output,
                status_output,
            ],
        )


demo.queue()


if __name__ == "__main__":
    demo.launch(head=COUNTDOWN_HEAD)
