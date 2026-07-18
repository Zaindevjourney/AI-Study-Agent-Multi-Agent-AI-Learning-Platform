"""
tutor_agent.py
--------------
Conversational explainer: "Explain X", "Explain in Urdu", "ELI10",
"give real-life examples", "compare A and B" - grounded in retrieved
PDF context when available.
"""

from typing import Optional

from llm_client import LLMClient
from database.vector_store import SimpleVectorStore


class TutorAgent:
    def __init__(self, llm: LLMClient, vector_store: SimpleVectorStore):
        self.llm = llm
        self.store = vector_store

    def ask(
        self,
        question: str,
        mode: str = "normal",
        language: str = "English",
        doc_id: Optional[str] = None,
        chat_history: Optional[list] = None,
    ) -> str:
        context = self._get_context(question, doc_id)

        mode_instructions = {
            "normal": "Explain clearly and accurately.",
            "eli10": "Explain as if the student is 10 years old - simple words, a fun analogy.",
            "examples": "Focus on giving 2-3 real-life examples to build intuition.",
            "compare": "Structure the answer as a clear comparison (similarities and differences).",
        }
        instruction = mode_instructions.get(mode, mode_instructions["normal"])

        system = (
            "You are the AI Tutor Agent inside a study assistant. "
            f"Respond in {language}. {instruction} "
            "Ground your answer in the provided source material when it's relevant; "
            "if the material doesn't cover it, answer from general knowledge and say so."
        )

        history_text = ""
        if chat_history:
            history_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[-6:])

        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Relevant source material:\n{context}\n\n"
            f"Student's question: {question}"
        )
        return self.llm.chat(prompt, system=system)

    def _get_context(self, question: str, doc_id: Optional[str], top_k: int = 4) -> str:
        results = self.store.search(question, top_k=top_k)
        if doc_id:
            results = [r for r in results if r["meta"].get("source_doc") == doc_id] or results
        if not results:
            return "(no matching source material found - answering from general knowledge)"
        return "\n\n---\n\n".join(r["text"] for r in results)
