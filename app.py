import logging
import random
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import spaces

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from filler_words import (
    format_filler_chips_html,
    highlight_fillers_html,
    summarize_fillers,
)
from rehearsal_limits import (
    MAX_RECORDING_SECONDS,
    accepted_recording_window_label,
    estimate_gpu_duration_seconds,
    validate_recording_duration_seconds,
)
from review import review_speech
from speech_stats import build_transcript_stats
from timing import (
    format_pacing_html,
    format_pacing_preview_html,
    get_audio_duration_seconds,
    summarize_timing,
)
from transcribe import transcribe_recording


APP_DIR = Path(__file__).parent
RECORDING_WINDOW_LABEL = accepted_recording_window_label()
SPEECH_FEEDBACK_PENDING = "_The honest review lands once the coach has heard you out._"
TIMING_FEEDBACK_PENDING = "_Pacing notes land after the transcript._"
FILLER_FEEDBACK_PENDING = '<div class="chip-empty">Crutch-word count lands after the transcript.</div>'
COMPLETION_STATUS = "All done — go raise that glass."

CONFETTI_COLORS = ("#bf8a3a", "#7a2636", "#183d34", "#fff9ec", "#efe3ce")
CONFETTI_PIECES = 28


def _build_confetti_html() -> str:
    rng = random.Random(7)
    pieces: list[str] = []
    for _ in range(CONFETTI_PIECES):
        left = rng.uniform(0, 100)
        color = rng.choice(CONFETTI_COLORS)
        delay = rng.uniform(0, 0.7)
        duration = rng.uniform(2.4, 3.8)
        drift = rng.uniform(-12, 12)
        style = (
            f"left:{left:.1f}vw;"
            f"background:{color};"
            f"--drift:{drift:.1f}vw;"
            f"animation-delay:{delay:.2f}s;"
            f"animation-duration:{duration:.2f}s;"
        )
        pieces.append(f'<span class="confetti-piece" style="{style}"></span>')
    return f'<div id="confetti-stage" aria-hidden="true">{"".join(pieces)}</div>'


HERO_RINGS_SVG = (
    '<svg class="hero-rings" viewBox="0 0 80 36" aria-hidden="true">'
    '<circle class="hero-rings__left" cx="28" cy="18" r="15"/>'
    '<circle class="hero-rings__right" cx="52" cy="18" r="15"/>'
    "</svg>"
)


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

  const watchCompletion = () => {{
    const status = document.querySelector("#status-output");
    const stage = document.querySelector("#confetti-stage");
    if (!status || !stage) {{
      window.setTimeout(watchCompletion, 500);
      return;
    }}

    const completionMarker = "All done";
    let lastTriggered = false;
    let resetTimer = null;

    const burst = () => {{
      stage.classList.remove("is-bursting");
      void stage.offsetWidth;
      stage.classList.add("is-bursting");
      if (resetTimer) {{
        window.clearTimeout(resetTimer);
      }}
      resetTimer = window.setTimeout(() => {{
        stage.classList.remove("is-bursting");
        resetTimer = null;
      }}, 4200);
    }};

    const observer = new MutationObserver(() => {{
      const text = status.textContent || "";
      const seen = text.includes(completionMarker);
      if (seen && !lastTriggered) {{
        burst();
      }}
      lastTriggered = seen;
    }});

    observer.observe(status, {{childList: true, characterData: true, subtree: true}});
  }};

  window.addEventListener("load", wireAudioButtons);
  window.addEventListener("load", watchCompletion);
  window.addEventListener("beforeunload", () => stopTimer());
}})();
</script>
"""

COUNTDOWN_HEAD += """
<script>
  (() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
      url.searchParams.set("__theme", "light");
      window.location.replace(url.toString());
    }
  })();
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
        highlight_fillers_html(transcript),
        feedback,
        timing_feedback,
        filler_feedback,
        _status_with_timings(status, timer, log=log_status),
    )


def _format_final_outputs(
    audio_path: str,
    transcript: str,
    feedback: str,
    timing_feedback: str,
    filler_feedback: str,
    status: str,
    timer: ProcessingTimer,
) -> tuple[str, str, str, str, str]:
    step_started_at = time.perf_counter()
    formatted_feedback = _format_speech_feedback_markdown(feedback)
    formatted_timing = format_pacing_html(audio_path, transcript)
    formatted_filler = format_filler_chips_html(transcript)
    timer.add_step("formatting", time.perf_counter() - step_started_at)

    return (
        highlight_fillers_html(transcript),
        formatted_feedback,
        formatted_timing,
        formatted_filler,
        _status_with_timings(status, timer),
    )


@spaces.GPU(duration=_gpu_duration_seconds)
def _process_valid_rehearsal(
    audio_path: str,
    duration_seconds: float,
    requested_gpu_seconds: int,
) -> Iterator[tuple[str, str, str, str, str]]:
    timer = ProcessingTimer(requested_gpu_seconds=requested_gpu_seconds)
    timing_preview = format_pacing_preview_html(duration_seconds)
    yield _outputs(
        "",
        SPEECH_FEEDBACK_PENDING,
        timing_preview,
        FILLER_FEEDBACK_PENDING,
        (
            f"Got the recording — {duration_seconds:.1f}s on the clock. "
            f"GPU budget reserved: {requested_gpu_seconds}s. Listening back now."
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
        "Transcript ready. Clocking your pacing.",
        timer,
    )
    timing_feedback = _timed_step(timer, "timing analysis", lambda: _build_timing_feedback(audio_path, transcript))
    formatted_timing_feedback = format_pacing_html(audio_path, transcript)

    yield _outputs(
        transcript,
        SPEECH_FEEDBACK_PENDING,
        formatted_timing_feedback,
        FILLER_FEEDBACK_PENDING,
        "Pacing logged. Counting the crutch words.",
        timer,
    )
    filler_feedback = _timed_step(timer, "filler analysis", lambda: _build_filler_feedback(transcript))
    formatted_filler_feedback = format_filler_chips_html(transcript)

    yield _outputs(
        transcript,
        SPEECH_FEEDBACK_PENDING,
        formatted_timing_feedback,
        formatted_filler_feedback,
        "Crutch words tallied. Asking the coach for an honest take.",
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
            audio_path,
            transcript,
            str(exc),
            timing_feedback,
            filler_feedback,
            "Heard you out — but the coach got stage fright.",
            timer,
        )
        return
    except RuntimeError as exc:
        yield _format_final_outputs(
            audio_path,
            transcript,
            str(exc),
            timing_feedback,
            filler_feedback,
            "Heard you out — but the coach got stage fright.",
            timer,
        )
        return
    except Exception as exc:
        yield _format_final_outputs(
            audio_path,
            transcript,
            f"Review failed: {exc}",
            timing_feedback,
            filler_feedback,
            "Heard you out — but the coach got stage fright.",
            timer,
        )
        return

    yield _format_final_outputs(
        audio_path,
        transcript,
        feedback,
        timing_feedback,
        filler_feedback,
        COMPLETION_STATUS,
        timer,
    )


def _clear_outputs(status: str) -> tuple[str, str, str, str, str]:
    return "", "", "", "", status


def _reset_rehearsal() -> tuple[None, str, str, str, str, str]:
    return (None, *_clear_outputs("Glass raised — ready when you are."))


def process_rehearsal(audio_path: str | None) -> Iterator[tuple[str, str, str, str, str]]:
    if not audio_path:
        yield _clear_outputs(f"Record a speech first. We accept run-throughs from {RECORDING_WINDOW_LABEL}.")
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
        gr.HTML(_build_confetti_html())
        gr.HTML(
            f"""
            <section id="hero-panel">
              <div class="kicker">Before the toast</div>
              {HERO_RINGS_SVG}
              <h1>Best Man Speech Coach</h1>
              <p>
                Two minutes at the mic and you'll get the toast back as a transcript,
                with pacing, crutch words, and a coach's honest note attached.
              </p>
              <div class="stamp">A two-minute rehearsal coach</div>
            </section>
            """
        )

        with gr.Row(elem_id="practice-grid"):
            with gr.Column(scale=7, elem_classes=["scorecard-card"]):
                gr.Markdown(
                    "## Step up to the mic\n"
                    f"Take a breath, then go. We need {RECORDING_WINDOW_LABEL} to give you a fair read."
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speech recording",
                    elem_id="speech-audio",
                    editable=False,
                    buttons=[],
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#bf8a3a",
                        waveform_progress_color="#7a2636",
                    ),
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
                gr.Markdown("## How it's going")
                status_output = gr.Markdown(
                    value="Glass raised — ready when you are.",
                    elem_id="status-output",
                )

        with gr.Row(elem_id="results-grid"):
            with gr.Column(scale=6, elem_classes=["scorecard-card", "result-panel"]):
                gr.Markdown("## What you said")
                transcript_output = gr.HTML(
                    value=(
                        '<div class="transcript-paper transcript-paper--empty">'
                        "Your transcript will land here once we've heard the recording."
                        "</div>"
                    ),
                    elem_id="transcript-output",
                )

            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Pacing")
                timing_output = gr.HTML(
                    value='<p class="pace-note">Pacing notes land here after the recording.</p>',
                    elem_id="timing-output",
                    elem_classes=["score-output"],
                )

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Crutch words")
                filler_output = gr.HTML(
                    value='<div class="chip-empty">Crutch-word count lands here after the recording.</div>',
                    elem_classes=["score-output"],
                )

            with gr.Column(scale=6, elem_classes=["scorecard-card", "result-panel"], elem_id="review-card"):
                gr.Markdown("## The honest review")
                feedback_output = gr.Markdown(
                    value="_The honest review will land here once the coach has heard you out._",
                    elem_classes=["score-output"],
                )

        transcribe_button.click(
            fn=process_rehearsal,
            inputs=audio_input,
            outputs=[transcript_output, feedback_output, timing_output, filler_output, status_output],
        )

        audio_input.change(
            fn=None,
            inputs=None,
            outputs=transcribe_button,
            js="() => ({ __type__: 'update', interactive: true })",
        )

        audio_input.clear(
            fn=None,
            inputs=None,
            outputs=transcribe_button,
            js="() => ({ __type__: 'update', interactive: false })",
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
        ).then(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=transcribe_button,
        )


demo.queue()


if __name__ == "__main__":
    demo.launch(head=COUNTDOWN_HEAD)
