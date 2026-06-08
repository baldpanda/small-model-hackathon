import gradio as gr


def greet_rehearsal(text: str) -> str:
    message = text.strip()
    if not message:
        return "Speech Coach is alive. Try typing a short rehearsal note."

    return f'Speech Coach is alive. You said: "{message}"'


demo = gr.Interface(
    fn=greet_rehearsal,
    inputs=gr.Textbox(
        label="Test message",
        placeholder="I'm testing the app.",
    ),
    outputs=gr.Textbox(label="App response"),
    title="Best Man Speech Coach",
    description="A tiny smoke test for the hackathon app.",
)


if __name__ == "__main__":
    demo.launch()
