"""
vector_store.py
----------------
A tiny, dependency-light vector store used for semantic-ish search over
PDF chunks. It builds simple TF-IDF vectors with numpy so it works fully
offline (no downloading of embedding models needed). Swap this out for
ChromaDB / FAISS / Pinecone in production - the interface (add, search)
is deliberately the same shape so that swap is a drop-in change.
"""

import re
import math
from collections import Counter
from typing import List, Dict

import numpy as np


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class SimpleVectorStore:
    def __init__(self):
        self.documents: List[Dict] = []   # [{id, text, meta}]
        self.vocab: Dict[str, int] = {}
        self.doc_freq: Counter = Counter()
        self._matrix = None  # cached tf-idf matrix

    def add(self, doc_id: str, text: str, meta: Dict = None):
        self.documents.append({"id": doc_id, "text": text, "meta": meta or {}})
        for token in set(_tokenize(text)):
            self.doc_freq[token] += 1
        self._matrix = None  # invalidate cache

    def _build_vocab(self):
        self.vocab = {tok: i for i, tok in enumerate(sorted(self.doc_freq.keys()))}

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        tokens = _tokenize(text)
        counts = Counter(tokens)
        n_docs = max(len(self.documents), 1)
        for tok, cnt in counts.items():
            if tok in self.vocab:
                idf = math.log((1 + n_docs) / (1 + self.doc_freq[tok])) + 1
                vec[self.vocab[tok]] = cnt * idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _ensure_matrix(self):
        if self._matrix is None:
            self._build_vocab()
            self._matrix = np.stack([self._vectorize(d["text"]) for d in self.documents]) \
                if self.documents else np.zeros((0, 0))

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.documents:
            return []
        self._ensure_matrix()
        q_vec = self._vectorize(query)
        if self._matrix.shape[1] != q_vec.shape[0]:
            return []
        sims = self._matrix @ q_vec
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            if sims[i] > 0:
                results.append({**self.documents[i], "score": float(sims[i])})
        return results

    def all_chunks(self) -> List[Dict]:
        return self.documents
