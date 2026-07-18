"""
demo.py
-------
Runs the whole multi-agent pipeline end-to-end from the command line,
without needing to start the FastAPI server. Works fully offline in
mock mode if no GEMINI_API_KEY / OPENAI_API_KEY is set.

Usage:
    cd backend
    python demo.py
"""

from database.db import init_db
from database.vector_store import SimpleVectorStore
from llm_client import LLMClient
from agents.orchestrator import Orchestrator
from agents.planner_agent import PlannerAgent


SAMPLE_TEXT = """
Artificial Intelligence (AI) is the field of computer science focused on
building systems that can perform tasks that normally require human
intelligence, such as understanding language, recognizing images, and
making decisions.

Machine Learning (ML) is a subset of AI where systems learn patterns
from data instead of being explicitly programmed with rules. A common
formula in ML is the Mean Squared Error: MSE = (1/n) * sum((y_i - y_hat_i)^2),
used to measure how far predictions are from actual values.

Deep Learning is a subset of Machine Learning that uses neural networks
with many layers to learn complex patterns, and powers most modern AI
systems such as large language models.
"""


def main():
    print("=" * 70)
    print("AI STUDY AGENT - DEMO RUN")
    print("=" * 70)

    init_db()
    orchestrator = Orchestrator()

    # Simulate a "PDF" being processed by feeding text directly into the
    # vector store the same way pdf_agent would (skips needing a real file).
    doc_id = "sample_ai_notes"
    chunks = orchestrator.pdf_agent._chunk(orchestrator.pdf_agent._clean(SAMPLE_TEXT))
    for i, chunk in enumerate(chunks):
        orchestrator.vector_store.add(f"{doc_id}::chunk_{i}", chunk, {"source_doc": doc_id})
    print(f"\n[PDF Agent] Indexed {len(chunks)} chunk(s) from sample content.\n")

    print("-" * 70)
    print("NOTES AGENT (bullet notes on 'Machine Learning')")
    print("-" * 70)
    notes = orchestrator.notes_agent.generate("Machine Learning", style="bullet", doc_id=doc_id)
    print(notes)

    print("\n" + "-" * 70)
    print("QUIZ AGENT (3 MCQs, medium difficulty)")
    print("-" * 70)
    quiz = orchestrator.quiz_agent.generate("Machine Learning basics", "mcq", "medium", 3, doc_id)
    print(quiz)

    print("\n" + "-" * 70)
    print("FLASHCARD AGENT (3 cards)")
    print("-" * 70)
    cards = orchestrator.flashcard_agent.generate("AI and ML key terms", "AI", 3, doc_id)
    print(cards)

    print("\n" + "-" * 70)
    print("AI TUTOR AGENT (ELI10)")
    print("-" * 70)
    answer = orchestrator.tutor_agent.ask("Explain Machine Learning", mode="eli10", doc_id=doc_id)
    print(answer)

    print("\n" + "-" * 70)
    print("STUDY PLANNER AGENT")
    print("-" * 70)
    planner = PlannerAgent()
    plan = planner.build_plan(
        subjects=["AI", "Python", "Math"],
        exam_date="2026-07-20",
        hours_per_day=3,
        weak_subjects=["Math"],
    )
    for day in plan[:5]:
        print(day)
    print(f"... ({len(plan)} total days planned)")

    print("\n" + "-" * 70)
    print("ORCHESTRATOR (single natural-language request)")
    print("-" * 70)
    result = orchestrator.handle_request(
        user_id="demo_user",
        message="Make short notes and 2 easy MCQs about Deep Learning",
        doc_id=doc_id,
    )
    print(result)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
