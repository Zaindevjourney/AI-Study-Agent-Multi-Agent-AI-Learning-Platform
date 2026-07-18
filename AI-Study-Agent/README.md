# 🎓 AI Study Agent — Multi-Agent Architecture (Python)

A working Python implementation of the multi-agent study-assistant
architecture: an **Orchestrator** that routes requests to specialist
agents (PDF, Notes, Quiz, Flashcards, Tutor, Planner, Reminders,
Analytics, Search, Web Research, Memory).

This is a complete, runnable starting point — not a toy. It runs fully
offline in **mock mode** with zero API keys, and switches to real answers
the moment you add a `GEMINI_API_KEY` or `OPENAI_API_KEY`.

## Quick start

```bash
cd AI-Study-Agent
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env      # then add your GEMINI_API_KEY or OPENAI_API_KEY (optional)
```

### Option A — try it instantly, no server needed
```bash
cd backend
python demo.py
```
This runs the whole pipeline (PDF → Notes → Quiz → Flashcards → Tutor →
Planner → Orchestrator) end-to-end and prints the results. Works even
with no API key set (you'll see clearly-labeled `[MOCK LLM RESPONSE]`
output instead of real AI text).

### Option B — run the API server
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Then open **http://localhost:8000/docs** for interactive Swagger docs
covering every endpoint below.

## Endpoints

| Endpoint                     | Agent            | Purpose |
|-------------------------------|------------------|---------|
| `POST /upload-pdf`            | PDF Agent        | Upload + extract + OCR + chunk + index a PDF |
| `POST /ask`                   | Orchestrator     | Natural-language request, auto-routed to the right agent(s) |
| `POST /notes`                 | Notes Agent      | short / detailed / bullet / important_topics / definitions / formula_sheet / summary |
| `POST /quiz`                  | Quiz Agent       | mcq / true_false / fill_in_blank / short_question / long_question, easy/medium/hard |
| `POST /flashcards/generate`   | Flashcard Agent  | Auto-generate front/back flashcards |
| `GET  /flashcards/due`        | Flashcard Agent  | Cards due today (spaced repetition) |
| `POST /flashcards/review`     | Flashcard Agent  | Submit a review (SM-2-lite scheduling) |
| `POST /tutor`                 | AI Tutor Agent   | Explain / ELI10 / examples / compare, any language |
| `POST /planner`               | Study Planner    | Day-by-day plan from exam date + subjects + hours |
| `POST /reminder/email`        | Reminder Agent   | Send (or log) a reminder email |
| `POST /analytics/session`     | Analytics Agent  | Log a study session |
| `GET  /analytics/summary`     | Analytics Agent  | Hours studied, weak/strong topics |
| `GET  /search`                | Search Agent     | Semantic-ish search over stored PDF chunks |

## Example: the doc's own example, end to end

> "Meri PDF se notes banao aur 20 MCQs bhi bana do."
> (Make notes from my PDF and also create 20 MCQs.)

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@mybook.pdf" -F "doc_id=mybook"

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Make short notes and 20 MCQs about chapter 3", "doc_id": "mybook"}'
```
The Orchestrator detects both "notes" and "quiz" intents in one message
and calls both agents, returning both results in a single response.

## Design notes / what's simplified vs. the full architecture doc

To keep this project **runnable with zero external services**, a few
pieces from the original architecture doc are implemented with
lightweight, swappable stand-ins:

| Doc mentions              | This project uses                          | To go to production |
|----------------------------|---------------------------------------------|----------------------|
| ChromaDB / FAISS / Pinecone | `SimpleVectorStore` (numpy TF-IDF, in `database/vector_store.py`) | Swap the class for a Chroma/FAISS client — same `.add()` / `.search()` interface |
| PostgreSQL + Redis         | SQLite (`database/db.py`)                    | Point SQLAlchemy at Postgres, add Redis for caching/queues |
| Gemini / GPT-4.1           | `LLMClient` (`llm_client.py`) — auto-detects Gemini/OpenAI keys, falls back to a labeled mock | Just set `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` |
| Web Research Agent         | `WebResearchAgent` takes a pluggable `search_fn` | Wire in Tavily / Serper / Bing Search API |
| Push notifications         | `ReminderAgent.send_push()` stub             | Wire in FCM / APNs |
| LangGraph / Flowise orchestration | Plain Python `Orchestrator` with keyword-based intent routing | Swap `_detect_intents` for an LLM-classifier call, or wrap agents as LangGraph nodes |

## Folder structure

```
AI-Study-Agent/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── pdf_agent.py
│   │   ├── notes_agent.py
│   │   ├── quiz_agent.py
│   │   ├── flashcard_agent.py
│   │   ├── tutor_agent.py
│   │   ├── planner_agent.py
│   │   ├── reminder_agent.py
│   │   ├── analytics_agent.py
│   │   ├── search_agent.py       # Search Agent + Web Research Agent
│   │   └── memory_agent.py
│   ├── database/
│   │   ├── db.py                 # SQLite persistence
│   │   └── vector_store.py       # lightweight offline vector store
│   ├── llm_client.py              # unified Gemini/OpenAI/mock client
│   ├── main.py                    # FastAPI app
│   └── demo.py                    # run the whole pipeline with no server
├── uploads/                        # uploaded PDFs land here
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap (matches the doc's phases)

- ✅ Phase 1: PDF upload + text extraction (`/upload-pdf`)
- ✅ Phase 2: RAG-style retrieval + tutor chat over uploaded PDFs (`/tutor`, `/search`)
- ✅ Phase 3: Notes, quiz, flashcard generation agents
- ✅ Phase 4: Study planner, reminders, analytics
- ✅ Phase 5: Multi-agent orchestration with memory (`Orchestrator` + `MemoryAgent`)
  — next step for full parity with the doc is swapping the keyword router
  for LangGraph and adding a real web-search provider.
