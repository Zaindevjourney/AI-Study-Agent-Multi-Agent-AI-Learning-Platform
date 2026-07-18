"""
search_agent.py
---------------
Search Agent: searches inside stored PDF chunks / notes via the vector store.

Web Research Agent: falls back to internet search when the answer isn't
found inside the user's own material, then answers with citations.
Uses whatever web-search capability is wired in (`search_fn`) - in this
standalone project that's a pluggable function so it can be swapped for
a real search API (Serper, Tavily, Bing, etc.) without touching callers.
"""

from typing import Callable, List, Dict, Optional

from database.vector_store import SimpleVectorStore
from llm_client import LLMClient


class SearchAgent:
    def __init__(self, vector_store: SimpleVectorStore):
        self.store = vector_store

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.store.search(query, top_k=top_k)


class WebResearchAgent:
    def __init__(self, llm: LLMClient, search_fn: Optional[Callable[[str], List[Dict]]] = None):
        self.llm = llm
        # search_fn(query) -> [{"title": ..., "url": ..., "snippet": ...}, ...]
        self.search_fn = search_fn

    def answer(self, query: str) -> Dict:
        if not self.search_fn:
            return {
                "answer": (
                    "Web search isn't wired up in this standalone project. "
                    "Plug a search function (e.g. Tavily/Serper/Bing API) into "
                    "WebResearchAgent(search_fn=...) to enable this."
                ),
                "citations": [],
            }

        results = self.search_fn(query)
        context = "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
            for i, r in enumerate(results)
        )
        system = (
            "You are the Web Research Agent. Answer using the search results below "
            "and cite sources inline like [1], [2]. If results don't answer the "
            "question, say so honestly."
        )
        prompt = f"Question: {query}\n\nSearch results:\n{context}"
        answer = self.llm.chat(prompt, system=system)
        return {"answer": answer, "citations": results}
