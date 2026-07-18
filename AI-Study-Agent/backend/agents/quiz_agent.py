"""
quiz_agent.py
-------------
Generates MCQs, True/False, Fill-in-the-blank, Short & Long questions
at Easy / Medium / Hard difficulty, as structured JSON.
"""

import json
from typing import List, Dict

from llm_client import LLMClient
from database.vector_store import SimpleVectorStore
from database import db

QUESTION_TYPES = ["mcq", "true_false", "fill_in_blank", "short_question", "long_question"]
DIFFICULTIES = ["easy", "medium", "hard"]


class QuizAgent:
    def __init__(self, llm: LLMClient, vector_store: SimpleVectorStore):
        self.llm = llm
        self.store = vector_store

    def generate(
        self,
        topic: str,
        question_type: str = "mcq",
        difficulty: str = "medium",
        num_questions: int = 5,
        doc_id: str = None,
    ) -> List[Dict]:
        question_type = question_type if question_type in QUESTION_TYPES else "mcq"
        difficulty = difficulty if difficulty in DIFFICULTIES else "medium"

        context = self._get_context(topic, doc_id)

        system = (
            "You are the Quiz Agent inside an AI study assistant. "
            "You output ONLY valid JSON, no prose, no markdown fences."
        )
        schema_hint = self._schema_hint(question_type)
        prompt = (
            f"Create {num_questions} {difficulty} difficulty '{question_type}' questions "
            f"about: {topic}\n\n"
            f"Base the questions strictly on this source material:\n{context}\n\n"
            f"Return a JSON array. Each item must look like:\n{schema_hint}"
        )

        raw = self.llm.chat(prompt, system=system, json_mode=True)
        return self._safe_parse(raw)

    def grade(self, subject: str, topic: str, answers: List[Dict]) -> Dict:
        """answers: [{"correct": bool}, ...] already graded client-side / by comparison."""
        total = len(answers)
        score = sum(1 for a in answers if a.get("correct"))
        db.log_quiz_result(subject, topic, score, total)
        return {"score": score, "total": total, "percentage": round(100 * score / total, 1) if total else 0}

    # ------------------------------------------------------------------ #
    def _get_context(self, topic: str, doc_id: str = None, top_k: int = 5) -> str:
        results = self.store.search(topic, top_k=top_k)
        if doc_id:
            results = [r for r in results if r["meta"].get("source_doc") == doc_id] or results
        if not results:
            return topic
        return "\n\n---\n\n".join(r["text"] for r in results)

    def _schema_hint(self, qtype: str) -> str:
        if qtype == "mcq":
            return '{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A", "explanation": "..."}'
        if qtype == "true_false":
            return '{"question": "...", "answer": true, "explanation": "..."}'
        if qtype == "fill_in_blank":
            return '{"question": "The capital of ___ is Paris.", "answer": "France"}'
        if qtype == "short_question":
            return '{"question": "...", "model_answer": "2-3 sentence answer"}'
        return '{"question": "...", "model_answer": "full paragraph(s) answer"}'

    def _safe_parse(self, raw: str) -> List[Dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json\n", "", 1)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            return [{"error": "Could not parse model output as JSON", "raw": raw}]
