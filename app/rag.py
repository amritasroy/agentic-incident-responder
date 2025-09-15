# app/rag.py  (pure-Python TF-IDF retriever)
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"

class SimpleRAG:
    def __init__(self):
        self.docs: List[str] = []
        self.ids: List[str] = []
        for p in sorted(KB_DIR.glob("*.md")):
            self.docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            self.ids.append(p.stem)
        # basic TF-IDF (you can tweak analyzer/stop_words/etc.)
        self.vectorizer = TfidfVectorizer(strip_accents="unicode")
        self.doc_mat = self.vectorizer.fit_transform(self.docs) if self.docs else None

    def __call__(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.docs or self.doc_mat is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.doc_mat).ravel()  # cosine for L2-normalized tf-idf
        idxs = sims.argsort()[::-1][:top_k]
        out = []
        for i in idxs:
            out.append({
                "id": self.ids[i],
                "snippet": self.docs[i][:300],
                "score": float(sims[i])
            })
        return out
