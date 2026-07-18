"""
flashcard_agent.py
------------------
Automatically creates front/back flashcards from source material and
schedules reviews using a simplified SM-2 spaced-repetition algorithm.
"""

import json
from typing import List, Dict

from llm_client import LLMClient
from database.vector_store import SimpleVectorStore
from database import db


class FlashcardAgent:
    def __init__(self, llm: LLMClient, vector_store: SimpleVectorStore):
        self.llm = llm
        self.store = vector_store

    def generate(self, topic: str, subject: str, num_cards: int = 8, doc_id: str = None) -> List[Dict]:
        context = self._get_context(topic, doc_id)

        system = (
            "You are the Flashcard Agent inside an AI study assistant. "
            "You output ONLY a valid JSON array of flashcards, no prose."
        )
        prompt = (
            f"Create {num_cards} flashcards about: {topic}\n\n"
            f"Source material:\n{context}\n\n"
            'Return format: [{"front": "question or term", "back": "answer or definition"}, ...]'
        )
        raw = self.llm.chat(prompt, system=system, json_mode=True)
        cards = self._safe_parse(raw)

        for card in cards:
            if "front" in card and "back" in card:
                db.add_flashcard(subject, card["front"], card["back"])

        return cards

    def get_due_cards(self, subject: str = None) -> List[Dict]:
        return db.due_flashcards(subject)

    def review_card(self, card_id: int, quality: int) -> Dict:
        """quality: 0 (total blank) .. 5 (perfect recall) - drives spaced repetition."""
        return db.update_flashcard_review(card_id, quality)

    # ------------------------------------------------------------------ #
    def _get_context(self, topic: str, doc_id: str = None, top_k: int = 5) -> str:
        results = self.store.search(topic, top_k=top_k)
        if doc_id:
            results = [r for r in results if r["meta"].get("source_doc") == doc_id] or results
        if not results:
            return topic
        return "\n\n---\n\n".join(r["text"] for r in results)

    def _safe_parse(self, raw: str) -> List[Dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json\n", "", 1)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [{"error": "Could not parse model output as JSON", "raw": raw}]
