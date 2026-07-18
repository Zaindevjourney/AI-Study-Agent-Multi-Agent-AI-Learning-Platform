"""
planner_agent.py
----------------
Builds a day-by-day study plan given an exam date, subjects, and daily
available hours. Pure algorithmic scheduling (no LLM required), with an
optional LLM pass to add human-friendly notes per day.
"""

from datetime import date, timedelta
from typing import List, Dict, Optional

from llm_client import LLMClient


class PlannerAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def build_plan(
        self,
        subjects: List[str],
        exam_date: str,          # "YYYY-MM-DD"
        hours_per_day: float,
        weak_subjects: Optional[List[str]] = None,
        revision_days_before_exam: int = 2,
    ) -> List[Dict]:
        today = date.today()
        exam = date.fromisoformat(exam_date)
        total_days = max((exam - today).days, 1)
        weak_subjects = weak_subjects or []

        # Give weak subjects extra weight so they get more days.
        weights = {s: (2 if s in weak_subjects else 1) for s in subjects}
        weight_sum = sum(weights.values())

        study_days = max(total_days - revision_days_before_exam, 1)
        plan = []
        subject_cycle = self._weighted_cycle(subjects, weights)

        for i in range(study_days):
            day = today + timedelta(days=i)
            subject = subject_cycle[i % len(subject_cycle)]
            plan.append({
                "date": day.isoformat(),
                "day_name": day.strftime("%A"),
                "subject": subject,
                "hours": hours_per_day,
            })

        for i in range(revision_days_before_exam):
            day = exam - timedelta(days=revision_days_before_exam - i)
            plan.append({
                "date": day.isoformat(),
                "day_name": day.strftime("%A"),
                "subject": "Revision (all subjects)",
                "hours": hours_per_day,
            })

        return plan

    def _weighted_cycle(self, subjects: List[str], weights: Dict[str, int]) -> List[str]:
        cycle = []
        for s in subjects:
            cycle.extend([s] * weights[s])
        return cycle or subjects
