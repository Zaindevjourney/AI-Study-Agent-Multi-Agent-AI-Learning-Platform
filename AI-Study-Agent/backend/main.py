"""
main.py
-------
FastAPI entry point for the AI Study Agent. Exposes one endpoint per
capability from the architecture doc, plus a single `/ask` endpoint that
goes through the Orchestrator (auto-routes to the right agent(s)).

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.
"""

import os
import shutil
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from database.db import init_db
from agents.orchestrator import Orchestrator
from agents.planner_agent import PlannerAgent
from agents.reminder_agent import ReminderAgent
from agents.analytics_agent import AnalyticsAgent

app = FastAPI(title="AI Study Agent", version="1.0.0")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

init_db()
orchestrator = Orchestrator()
planner_agent = PlannerAgent()
reminder_agent = ReminderAgent()
analytics_agent = AnalyticsAgent()

# Serve a minimal static frontend (open /app in your browser)
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir), name="frontend")


@app.get('/app')
def _app_index():
    return RedirectResponse(url='/app/index.html')


# ---------------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------------- #
class AskRequest(BaseModel):
    user_id: str = "demo_user"
    message: str
    doc_id: Optional[str] = None
    subject: str = "General"


class NotesRequest(BaseModel):
    topic: str
    style: str = "short"
    doc_id: Optional[str] = None


class QuizRequest(BaseModel):
    topic: str
    question_type: str = "mcq"
    difficulty: str = "medium"
    num_questions: int = 5
    doc_id: Optional[str] = None


class FlashcardRequest(BaseModel):
    topic: str
    subject: str = "General"
    num_cards: int = 8
    doc_id: Optional[str] = None


class FlashcardReview(BaseModel):
    card_id: int
    quality: int  # 0-5


class TutorRequest(BaseModel):
    question: str
    mode: str = "normal"
    language: str = "English"
    doc_id: Optional[str] = None


class PlannerRequest(BaseModel):
    subjects: List[str]
    exam_date: str  # YYYY-MM-DD
    hours_per_day: float
    weak_subjects: Optional[List[str]] = None


class ReminderRequest(BaseModel):
    to_email: str
    subject: str
    body: str


class StudySessionRequest(BaseModel):
    subject: str
    minutes: int


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "message": "AI Study Agent backend is running."}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), doc_id: Optional[str] = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = orchestrator.pdf_agent.process(save_path, doc_id)
    return {"doc_id": doc_id, **result}


@app.post("/ask")
def ask(req: AskRequest):
    return orchestrator.handle_request(
        user_id=req.user_id, message=req.message, doc_id=req.doc_id, subject=req.subject
    )


@app.post("/notes")
def notes(req: NotesRequest):
    return {"notes": orchestrator.notes_agent.generate(req.topic, req.style, req.doc_id)}


@app.post("/quiz")
def quiz(req: QuizRequest):
    return {"questions": orchestrator.quiz_agent.generate(
        req.topic, req.question_type, req.difficulty, req.num_questions, req.doc_id
    )}


@app.post("/flashcards/generate")
def generate_flashcards(req: FlashcardRequest):
    return {"flashcards": orchestrator.flashcard_agent.generate(
        req.topic, req.subject, req.num_cards, req.doc_id
    )}


@app.get("/flashcards/due")
def due_flashcards(subject: Optional[str] = None):
    return {"due": orchestrator.flashcard_agent.get_due_cards(subject)}


@app.post("/flashcards/review")
def review_flashcard(req: FlashcardReview):
    result = orchestrator.flashcard_agent.review_card(req.card_id, req.quality)
    if result is None:
        raise HTTPException(404, "Flashcard not found.")
    return result


@app.post("/tutor")
def tutor(req: TutorRequest):
    return {"answer": orchestrator.tutor_agent.ask(req.question, req.mode, req.language, req.doc_id)}


@app.post("/planner")
def planner(req: PlannerRequest):
    plan = planner_agent.build_plan(req.subjects, req.exam_date, req.hours_per_day, req.weak_subjects)
    return {"plan": plan}


@app.post("/reminder/email")
def reminder_email(req: ReminderRequest):
    status = reminder_agent.send_email(req.to_email, req.subject, req.body)
    return {"status": status}


@app.post("/analytics/session")
def log_session(req: StudySessionRequest):
    analytics_agent.log_session(req.subject, req.minutes)
    return {"logged": True}


@app.get("/analytics/summary")
def analytics_summary():
    return analytics_agent.summary()


@app.get("/search")
def search(query: str):
    return {"results": orchestrator.search_agent.search(query)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
