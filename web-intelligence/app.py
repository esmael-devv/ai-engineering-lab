
import gradio as gr
import requests

from openai import OpenAIError
from summarizer import summarize_website


def generate_summary(url):
    if not url or not url.strip():
        return "Please enter a website URL."

    try:
        return summarize_website(url.strip())
    except (requests.RequestException, ValueError, OpenAIError) as error:
        return f"Unable to summarize website: {error}"


demo = gr.Interface(
    fn=generate_summary,
    inputs=gr.Textbox(
        label="Website URL",
        placeholder="https://www.python.org"
    ),
    outputs=gr.Markdown(label="AI Summary"),
    title="AI Website Intelligence",
    description=(
        "Extract and summarize website content "
        "using a Groq-hosted language model."
    ),
    flagging_mode="never"
)


if __name__ == "__main__":
    demo.launch(share=False)
