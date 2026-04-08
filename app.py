import os
import gradio as gr
from dotenv import load_dotenv
from rag.pipeline import build_pipeline

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        pipeline = build_pipeline(GROQ_API_KEY)
    return pipeline


def answer_question(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a cybersecurity question to get started."

    try:
        bot = get_pipeline()
        return bot.ask(question.strip())
    except Exception as e:
        return f"Error: {e}"


EXAMPLE_QUESTIONS = [
    ["What are the most common techniques used by ransomware groups to gain initial access?"],
    ["What does NIST recommend for incident response handling?"],
    ["What is the MITRE ATT&CK framework used for?"],
]

with gr.Blocks(title="CyberBot — RAG Cybersecurity Knowledge Assistant") as demo:
    gr.Markdown(
        """
# CyberBot — RAG Cybersecurity Knowledge Assistant

Ask questions about cybersecurity threats, vulnerabilities, and incident response.
Answers are grounded in real threat intelligence documents including CrowdStrike, NIST, MITRE ATT&CK, and WEF reports.
        """
    )

    with gr.Row():
        with gr.Column():
            question_box = gr.Textbox(
                lines=3,
                placeholder="e.g. What are the most common ransomware initial access techniques?",
                label="Your Question",
            )
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            response_box = gr.Textbox(
                lines=20,
                label="CyberBot Response",
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        outputs=response_box,
        fn=answer_question,
        run_on_click=True,
        label="Example Questions",
    )

    submit_btn.click(fn=answer_question, inputs=question_box, outputs=response_box)
    question_box.submit(fn=answer_question, inputs=question_box, outputs=response_box)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
    )
