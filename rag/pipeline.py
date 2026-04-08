from groq import Groq
from rag.retriever import Retriever


SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"


def _load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return (
            "You are a cybersecurity expert assistant. "
            "Answer questions accurately based only on the provided context."
        )


class Pipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.retriever = Retriever()
        self.system_prompt = _load_system_prompt()
        self.model = "llama-3.1-8b-instant"

    def ask(self, question: str) -> str:
        chunks = self.retriever.retrieve(question)

        if chunks:
            context = "\n\n---\n\n".join(chunks)
            user_message = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            user_message = (
                f"No relevant context was found in the knowledge base.\n\n"
                f"Question: {question}"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()


def build_pipeline(groq_api_key: str) -> Pipeline:
    return Pipeline(groq_api_key=groq_api_key)
