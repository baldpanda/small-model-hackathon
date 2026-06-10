import gradio as gr
import spaces

from review import review_speech
from transcribe import MAX_RECORDING_SECONDS, transcribe_recording


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


@spaces.GPU(duration=120)
def process_rehearsal(audio_path: str | None) -> tuple[str, str, str]:
    if not audio_path:
        return "", "", "Record a speech first. The app accepts up to 60 seconds."

    try:
        transcript = transcribe_recording(audio_path)
    except ValueError as exc:
        return "", "", str(exc)
    except RuntimeError as exc:
        return "", "", str(exc)
    except Exception as exc:
        return "", "", f"Transcription failed: {exc}"

    try:
        feedback = review_speech(transcript)
    except ValueError as exc:
        return transcript, str(exc), "Transcription complete. Review failed."
    except RuntimeError as exc:
        return transcript, str(exc), "Transcription complete. Review failed."
    except Exception as exc:
        return transcript, f"Review failed: {exc}", "Transcription complete. Review failed."

    return transcript, feedback, "Transcription and review complete."


with gr.Blocks(title="Best Man Speech Coach") as demo:
    gr.Markdown("# Best Man Speech Coach")
    gr.Markdown(
        "Record up to one minute of rehearsal audio, transcribe it with Cohere Transcribe, "
        "then review the transcript with OpenBMB MiniCPM5."
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
    transcribe_button = gr.Button("Review speech", variant="primary")

    transcript_output = gr.Textbox(
        label="Transcript",
        lines=10,
        placeholder="Your transcript will appear here after recording.",
    )
    feedback_output = gr.Textbox(
        label="Speech feedback",
        lines=12,
        placeholder="Speech feedback will appear here after transcription.",
    )
    status_output = gr.Textbox(
        label="Status",
        lines=2,
        interactive=False,
        value="Ready to record.",
    )

    transcribe_button.click(
        fn=process_rehearsal,
        inputs=audio_input,
        outputs=[transcript_output, feedback_output, status_output],
    )


if __name__ == "__main__":
    demo.launch(head=COUNTDOWN_HEAD)
