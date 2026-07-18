"""
db.py
-----
Minimal SQLite persistence layer (stand-in for the PostgreSQL + Redis
combo mentioned in the architecture doc). Keeps the project runnable
with zero external services while preserving the same responsibilities:
storing flashcards, quiz results, and analytics events.
"""

import sqlite3
import json
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "study_agent.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            front TEXT,
            back TEXT,
            ease REAL DEFAULT 2.5,
            interval_days INTEGER DEFAULT 1,
            due_at REAL,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            topic TEXT,
            score REAL,
            total INTEGER,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            minutes INTEGER,
            created_at REAL
        );
        """
    )
    conn.commit()
    conn.close()


def add_flashcard(subject, front, back):
    conn = get_conn()
    conn.execute(
        "INSERT INTO flashcards (subject, front, back, due_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (subject, front, back, time.time(), time.time()),
    )
    conn.commit()
    conn.close()


def due_flashcards(subject=None):
    conn = get_conn()
    now = time.time()
    if subject:
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE due_at <= ? AND subject = ? ORDER BY due_at",
            (now, subject),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE due_at <= ? ORDER BY due_at", (now,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_flashcard_review(card_id, quality: int):
    """SM-2-lite spaced repetition scheduler. quality: 0-5 (self-rated recall)."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM flashcards WHERE id = ?", (card_id,)).fetchone()
    if not row:
        conn.close()
        return None

    ease = row["ease"]
    interval = row["interval_days"]

    if quality < 3:
        interval = 1
    else:
        ease = max(1.3, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        interval = 1 if interval <= 1 else round(interval * ease)

    due_at = time.time() + interval * 86400
    conn.execute(
        "UPDATE flashcards SET ease = ?, interval_days = ?, due_at = ? WHERE id = ?",
        (ease, interval, due_at, card_id),
    )
    conn.commit()
    conn.close()
    return {"ease": ease, "interval_days": interval}


def log_quiz_result(subject, topic, score, total):
    conn = get_conn()
    conn.execute(
        "INSERT INTO quiz_results (subject, topic, score, total, created_at) VALUES (?, ?, ?, ?, ?)",
        (subject, topic, score, total, time.time()),
    )
    conn.commit()
    conn.close()


def log_study_session(subject, minutes):
    conn = get_conn()
    conn.execute(
        "INSERT INTO study_sessions (subject, minutes, created_at) VALUES (?, ?, ?)",
        (subject, minutes, time.time()),
    )
    conn.commit()
    conn.close()


def analytics_summary():
    conn = get_conn()
    quiz_rows = conn.execute(
        "SELECT subject, topic, AVG(score * 1.0 / total) as avg_pct, COUNT(*) as attempts "
        "FROM quiz_results GROUP BY subject, topic"
    ).fetchall()
    total_minutes = conn.execute(
        "SELECT COALESCE(SUM(minutes), 0) as total FROM study_sessions"
    ).fetchone()["total"]
    conn.close()

    topics = [dict(r) for r in quiz_rows]
    weak = sorted([t for t in topics if t["avg_pct"] is not None], key=lambda t: t["avg_pct"])[:3]
    strong = sorted([t for t in topics if t["avg_pct"] is not None], key=lambda t: -t["avg_pct"])[:3]

    return {
        "total_minutes_studied": total_minutes,
        "weak_topics": weak,
        "strong_topics": strong,
        "all_topics": topics,
    }
