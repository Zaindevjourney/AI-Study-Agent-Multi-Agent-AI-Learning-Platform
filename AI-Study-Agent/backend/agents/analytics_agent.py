"""
analytics_agent.py
-------------------
Tracks hours studied, quiz scores, weak/strong topics, and learning streaks.
"""

from database import db


class AnalyticsAgent:
    def log_session(self, subject: str, minutes: int):
        db.log_study_session(subject, minutes)

    def summary(self) -> dict:
        return db.analytics_summary()
