import os
import sys
from dotenv import load_dotenv
from rag.pipeline import build_pipeline

load_dotenv()


def main():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY is not set. Add it as a secret or in a .env file.")
        sys.exit(1)

    print("CyberBot — RAG Cybersecurity Assistant")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    pipeline = build_pipeline(groq_api_key)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        answer = pipeline.ask(question)
        print(f"\nCyberBot: {answer}\n")


if __name__ == "__main__":
    main()
