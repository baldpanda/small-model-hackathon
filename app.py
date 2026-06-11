from pathlib import Path

import gradio as gr
import spaces

from filler_words import summarize_fillers
from review import review_speech
from timing import summarize_timing
from transcribe import MAX_RECORDING_SECONDS, transcribe_recording


APP_DIR = Path(__file__).parent

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
        stopTimer("Reached 1:00. Recording stopped. Transcribe when ready.");
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
            stopTimer("Recording stopped. Transcribe when ready.");
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


@spaces.GPU(duration=120)
def process_rehearsal(audio_path: str | None) -> tuple[str, str, str, str, str]:
    if not audio_path:
        return "", "", "", "", "Record a speech first. The app accepts up to 60 seconds."

    try:
        transcript = transcribe_recording(audio_path)
    except ValueError as exc:
        return "", "", "", "", str(exc)
    except RuntimeError as exc:
        return "", "", "", "", str(exc)
    except Exception as exc:
        return "", "", "", "", f"Transcription failed: {exc}"

    timing_feedback = _build_timing_feedback(audio_path, transcript)
    filler_feedback = _build_filler_feedback(transcript)

    try:
        feedback = review_speech(transcript)
    except ValueError as exc:
        return (
            transcript,
            _format_speech_feedback_markdown(str(exc)),
            _format_metric_markdown(timing_feedback),
            _format_metric_markdown(filler_feedback),
            "Transcription complete. Review failed.",
        )
    except RuntimeError as exc:
        return (
            transcript,
            _format_speech_feedback_markdown(str(exc)),
            _format_metric_markdown(timing_feedback),
            _format_metric_markdown(filler_feedback),
            "Transcription complete. Review failed.",
        )
    except Exception as exc:
        return (
            transcript,
            _format_speech_feedback_markdown(f"Review failed: {exc}"),
            _format_metric_markdown(timing_feedback),
            _format_metric_markdown(filler_feedback),
            "Transcription complete. Review failed.",
        )

    return (
        transcript,
        _format_speech_feedback_markdown(feedback),
        _format_metric_markdown(timing_feedback),
        _format_metric_markdown(filler_feedback),
        "Transcription, review, timing, and filler analysis complete.",
    )


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
            """
            <section id="hero-panel">
              <div class="kicker">Build Small Hackathon rehearsal desk</div>
              <h1>Best Man Speech Coach</h1>
              <p>
                Record a one-minute run-through and get a wedding-scorecard readout:
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
                    "Speak naturally. The recording cap keeps each pass short enough to retry."
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speech recording",
                    elem_id="speech-audio",
                )
                countdown = gr.HTML(
                    f"<div id='recording-status'>Recording limit: 1:00</div>",
                    label="Recording timer",
                )
                transcribe_button = gr.Button(
                    "Review speech",
                    variant="primary",
                    elem_id="review-button",
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

            with gr.Column(scale=6, elem_classes=["scorecard-card", "result-panel"]):
                gr.Markdown("## Speech Feedback")
                feedback_output = gr.Markdown(
                    value="_Speech feedback will appear here after transcription._",
                    elem_classes=["score-output"],
                )

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Timing Feedback")
                timing_output = gr.Markdown(
                    value="_Timing feedback will appear here after transcription._",
                    elem_id="timing-output",
                    elem_classes=["score-output"],
                )

            with gr.Column(scale=6, elem_classes=["scorecard-card", "metric-panel"]):
                gr.Markdown("## Filler Feedback")
                filler_output = gr.Markdown(
                    value="_Filler feedback will appear here after transcription._",
                    elem_classes=["score-output"],
                )

        transcribe_button.click(
            fn=process_rehearsal,
            inputs=audio_input,
            outputs=[transcript_output, feedback_output, timing_output, filler_output, status_output],
        )


if __name__ == "__main__":
    demo.launch(head=COUNTDOWN_HEAD)
