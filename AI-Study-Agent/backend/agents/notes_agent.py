"""
notes_agent.py
--------------
Generates: short notes, detailed notes, bullet notes, important topics,
definitions, formula sheet, and summaries from retrieved PDF context.
"""

from llm_client import LLMClient
from database.vector_store import SimpleVectorStore

NOTE_STYLES = {
    "short": "Write short, exam-friendly notes (max 200 words).",
    "detailed": "Write detailed, in-depth study notes covering every concept thoroughly.",
    "bullet": "Write the notes as clean, nested bullet points only.",
    "important_topics": "List only the most important topics/headings a student must revise, one line each.",
    "definitions": "Extract and clearly define every key term / definition found in the content.",
    "formula_sheet": "Extract every formula found in the content into a clean formula sheet, with a one-line explanation of each variable.",
    "summary": "Write a concise summary (3-5 paragraphs) of the content.",
}


class NotesAgent:
    def __init__(self, llm: LLMClient, vector_store: SimpleVectorStore):
        self.llm = llm
        self.store = vector_store

    def generate(self, topic: str, style: str = "short", doc_id: str = None) -> str:
        style = style if style in NOTE_STYLES else "short"
        context = self._get_context(topic, doc_id)

        system = (
            "You are the Notes Agent inside an AI study assistant. "
            "You turn source material into clear, student-friendly study notes."
        )
        prompt = (
            f"Instruction: {NOTE_STYLES[style]}\n\n"
            f"Topic requested: {topic}\n\n"
            f"Source material:\n{context}\n\n"
            "Only use information supported by the source material above."
        )
        return self.llm.chat(prompt, system=system)

    def _get_context(self, topic: str, doc_id: str = None, top_k: int = 5) -> str:
        results = self.store.search(topic, top_k=top_k)
        if doc_id:
            results = [r for r in results if r["meta"].get("source_doc") == doc_id] or results
        if not results:
            return topic  # fall back to using the topic text itself
        return "\n\n---\n\n".join(r["text"] for r in results)
