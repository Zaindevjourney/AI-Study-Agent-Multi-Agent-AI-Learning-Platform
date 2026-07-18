"""
llm_client.py
--------------
A single, unified interface for talking to an LLM so every agent
just calls `llm.chat(prompt)` without caring which provider is behind it.

Supported providers (auto-detected from environment variables):
    - Gemini  (set GEMINI_API_KEY)
    - OpenAI  (set OPENAI_API_KEY)
    - Mock    (used automatically if no key is found, so the whole
               project still runs end-to-end for demos / testing)

Usage:
    from llm_client import LLMClient
    llm = LLMClient()
    reply = llm.chat("Explain photosynthesis in simple words")
"""

import os
import json
import textwrap
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if provider:
            self.provider = provider
        elif self.gemini_key:
            self.provider = "gemini"
        elif self.openai_key:
            self.provider = "openai"
        else:
            self.provider = "mock"

        self.model = model or {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4.1",
            "mock": "mock-llm",
        }[self.provider]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        """Send a prompt to whichever provider is configured and return text."""
        if self.provider == "gemini":
            return self._call_gemini(prompt, system, json_mode)
        if self.provider == "openai":
            return self._call_openai(prompt, system, json_mode)
        return self._call_mock(prompt, system, json_mode)

    # ------------------------------------------------------------------ #
    # Providers
    # ------------------------------------------------------------------ #
    def _call_gemini(self, prompt, system, json_mode):
        import requests
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.gemini_key}"
        )
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        if json_mode:
            payload["generationConfig"] = {"response_mime_type": "application/json"}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt, system, json_mode):
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers, timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _call_mock(self, prompt, system, json_mode):
        """
        No API key configured -> return a clearly-labeled placeholder so the
        rest of the pipeline (chunking, storage, routing, etc.) can still be
        demonstrated and tested without network access or billing.
        """
        preview = textwrap.shorten(prompt, width=120)
        if json_mode:
            return json.dumps({
                "mock": True,
                "note": "Set GEMINI_API_KEY or OPENAI_API_KEY for real answers.",
                "prompt_preview": preview,
            })
        return (
            "[MOCK LLM RESPONSE - set GEMINI_API_KEY or OPENAI_API_KEY in your "
            f".env for real output]\n\nPrompt received: {preview}"
        )
