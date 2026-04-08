import os
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


def _format_sources(metadatas: list[dict]) -> str:
    source_pages: dict[str, list[int]] = {}

    for meta in metadatas:
        raw_source = meta.get("source", "")
        filename = os.path.basename(raw_source) if raw_source else "unknown"
        page = meta.get("page")

        if filename not in source_pages:
            source_pages[filename] = []

        if page is not None:
            try:
                page_num = int(page) + 1
                if page_num not in source_pages[filename]:
                    source_pages[filename].append(page_num)
            except (ValueError, TypeError):
                pass

    lines = []
    for filename, pages in source_pages.items():
        if pages:
            pages.sort()
            page_str = ", ".join(str(p) for p in pages)
            lines.append(f"- {filename} (page {page_str})")
        else:
            lines.append(f"- {filename}")

    return "Sources:\n" + "\n".join(lines)


class Pipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.retriever = Retriever()
        self.system_prompt = _load_system_prompt()
        self.model = "llama-3.1-8b-instant"

    def ask(self, question: str) -> str:
        chunks, metadatas = self.retriever.retrieve(question)

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

        answer = response.choices[0].message.content.strip()

        # Strip any "Sources:" block the LLM may have added itself to prevent duplicates
        if "\nSources:" in answer:
            answer = answer[:answer.index("\nSources:")].strip()

        # Detect fallback response — per system prompt, no sources on fallback
        is_fallback = answer.startswith("That specific information is not available")

        if metadatas and not is_fallback:
            sources = _format_sources(metadatas)
            return f"{answer}\n\n{sources}"

        return answer


def build_pipeline(groq_api_key: str) -> Pipeline:
    return Pipeline(groq_api_key=groq_api_key)
