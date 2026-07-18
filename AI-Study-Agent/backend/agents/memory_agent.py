"""
memory_agent.py
---------------
Remembers per-user context across turns: previous conversations, subjects,
weak chapters, preferred language, exam dates. This demo keeps memory
in a simple JSON file per user_id (swap for Redis/Postgres in production).
"""

import json
import os
import time

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memory_store")


class MemoryAgent:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)

    def _path(self, user_id: str) -> str:
        return os.path.join(MEMORY_DIR, f"{user_id}.json")

    def load(self, user_id: str) -> dict:
        path = self._path(user_id)
        if not os.path.exists(path):
            return {
                "user_id": user_id,
                "preferred_language": "English",
                "subjects": [],
                "weak_chapters": [],
                "exam_dates": {},
                "chat_history": [],
            }
        with open(path, "r") as f:
            return json.load(f)

    def save(self, user_id: str, memory: dict):
        with open(self._path(user_id), "w") as f:
            json.dump(memory, f, indent=2)

    def remember_message(self, user_id: str, role: str, content: str):
        memory = self.load(user_id)
        memory["chat_history"].append({"role": role, "content": content, "ts": time.time()})
        memory["chat_history"] = memory["chat_history"][-50:]  # cap history
        self.save(user_id, memory)

    def set_preference(self, user_id: str, key: str, value):
        memory = self.load(user_id)
        memory[key] = value
        self.save(user_id, memory)
