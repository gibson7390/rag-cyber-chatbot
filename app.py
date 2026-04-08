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


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        lines=3,
        placeholder="e.g. What are the most common ransomware initial access techniques?",
        label="Your Question",
    ),
    outputs=gr.Textbox(
        lines=20,
        label="CyberBot Response",
    ),
    title="CyberBot — RAG Cybersecurity Knowledge Assistant",
    description=(
        "Ask questions about cybersecurity threats, vulnerabilities, and incident response. "
        "Answers are grounded in real threat intelligence documents including CrowdStrike, "
        "NIST, MITRE ATT&CK, and WEF reports."
    ),
    examples=[
        ["What are the most common techniques used by ransomware groups to gain initial access?"],
        ["What does NIST recommend for incident response handling?"],
        ["What is the MITRE ATT&CK framework used for?"],
        ["What are the top cyber threats facing organizations in 2025 and 2026?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
    )
