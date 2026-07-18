"""
orchestrator.py
----------------
The Main AI Orchestrator Agent. Understands the user's request and routes
it to the right sub-agent(s), e.g.:

    "Meri PDF se notes banao aur 20 MCQs bhi bana do."
    -> PDF Agent (if a new file) -> Notes Agent -> Quiz Agent -> save results

Routing is done with lightweight keyword rules by default (fast, free,
deterministic) with an optional LLM-based intent classifier for messier
natural-language requests.
"""

import json
from typing import Dict, List, Optional

from llm_client import LLMClient
from database.vector_store import SimpleVectorStore

from agents.pdf_agent import PDFAgent
from agents.notes_agent import NotesAgent
from agents.quiz_agent import QuizAgent
from agents.flashcard_agent import FlashcardAgent
from agents.tutor_agent import TutorAgent
from agents.planner_agent import PlannerAgent
from agents.reminder_agent import ReminderAgent
from agents.analytics_agent import AnalyticsAgent
from agents.search_agent import SearchAgent, WebResearchAgent
from agents.memory_agent import MemoryAgent


class Orchestrator:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        self.vector_store = SimpleVectorStore()

        self.pdf_agent = PDFAgent(self.vector_store)
        self.notes_agent = NotesAgent(self.llm, self.vector_store)
        self.quiz_agent = QuizAgent(self.llm, self.vector_store)
        self.flashcard_agent = FlashcardAgent(self.llm, self.vector_store)
        self.tutor_agent = TutorAgent(self.llm, self.vector_store)
        self.planner_agent = PlannerAgent(self.llm)
        self.reminder_agent = ReminderAgent()
        self.analytics_agent = AnalyticsAgent()
        self.search_agent = SearchAgent(self.vector_store)
        self.web_research_agent = WebResearchAgent(self.llm)
        self.memory_agent = MemoryAgent()

    # ------------------------------------------------------------------ #
    def handle_request(self, user_id: str, message: str, pdf_path: Optional[str] = None,
                        doc_id: Optional[str] = None, subject: str = "General") -> Dict:
        """
        Single entry point mirroring the doc's example:
        "Meri PDF se notes banao aur 20 MCQs bhi bana do." (Make notes from
        my PDF and also create 20 MCQs.)
        """
        self.memory_agent.remember_message(user_id, "user", message)
        results = {}

        # 1) New PDF? -> run PDF Agent first so downstream agents have context.
        if pdf_path:
            results["pdf_agent"] = self.pdf_agent.process(pdf_path, doc_id or "uploaded_doc")
            doc_id = doc_id or "uploaded_doc"

        intents = self._detect_intents(message)

        if "notes" in intents:
            results["notes_agent"] = self.notes_agent.generate(
                topic=message, style=intents.get("notes_style", "short"), doc_id=doc_id
            )

        if "quiz" in intents:
            results["quiz_agent"] = self.quiz_agent.generate(
                topic=message,
                question_type=intents.get("quiz_type", "mcq"),
                difficulty=intents.get("difficulty", "medium"),
                num_questions=intents.get("num_questions", 5),
                doc_id=doc_id,
            )

        if "flashcards" in intents:
            results["flashcard_agent"] = self.flashcard_agent.generate(
                topic=message, subject=subject, doc_id=doc_id
            )

        if "tutor" in intents:
            results["tutor_agent"] = self.tutor_agent.ask(message, mode=intents.get("tutor_mode", "normal"), doc_id=doc_id)

        if "planner" in intents:
            results["planner_agent"] = "Use /planner endpoint with exam_date, subjects, hours_per_day."

        if "search" in intents:
            results["search_agent"] = self.search_agent.search(message)

        if not results:
            # Default: treat as a tutor question.
            results["tutor_agent"] = self.tutor_agent.ask(message, doc_id=doc_id)

        self.memory_agent.remember_message(user_id, "assistant", json.dumps(results, default=str)[:2000])
        return results

    # ------------------------------------------------------------------ #
    def _detect_intents(self, message: str) -> Dict:
        """Fast keyword-based router. Swap for an LLM classifier if desired."""
        m = message.lower()
        intents: Dict = {}

        if any(w in m for w in ["note", "notes", "summary", "summarize"]):
            intents["notes"] = True
            if "bullet" in m:
                intents["notes_style"] = "bullet"
            elif "detailed" in m:
                intents["notes_style"] = "detailed"
            elif "formula" in m:
                intents["notes_style"] = "formula_sheet"
            elif "definition" in m:
                intents["notes_style"] = "definitions"
            elif "important" in m:
                intents["notes_style"] = "important_topics"

        if any(w in m for w in ["mcq", "quiz", "question", "test me"]):
            intents["quiz"] = True
            if "mcq" in m or "multiple choice" in m:
                intents["quiz_type"] = "mcq"
            elif "true" in m and "false" in m:
                intents["quiz_type"] = "true_false"
            elif "fill" in m:
                intents["quiz_type"] = "fill_in_blank"
            elif "long question" in m or "long answer" in m:
                intents["quiz_type"] = "long_question"
            elif "short question" in m or "short answer" in m:
                intents["quiz_type"] = "short_question"
            for word in m.split():
                if word.isdigit():
                    intents["num_questions"] = int(word)
                    break
            for diff in ["easy", "medium", "hard"]:
                if diff in m:
                    intents["difficulty"] = diff

        if "flashcard" in m or "flash card" in m:
            intents["flashcards"] = True

        if any(w in m for w in ["explain", "eli", "like i'm", "compare", "example"]):
            intents["tutor"] = True
            if "urdu" in m:
                intents["tutor_mode"] = "normal"  # language handled separately
            if "10" in m or "child" in m:
                intents["tutor_mode"] = "eli10"
            if "compare" in m:
                intents["tutor_mode"] = "compare"
            if "example" in m:
                intents["tutor_mode"] = "examples"

        if any(w in m for w in ["study plan", "planner", "schedule"]):
            intents["planner"] = True

        if any(w in m for w in ["search my", "find in my notes", "find in pdf"]):
            intents["search"] = True

        return intents
