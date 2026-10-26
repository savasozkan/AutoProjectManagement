"""Find existing project tasks that are similar to a new document.

The document is split into chunks; each task is scored by its best-matching chunk, so a task that
covers one small part of a long document still scores well. Scoring uses an Ollama embedding
model when available and falls back to word overlap (TF-IDF) otherwise.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .ollama import OllamaClient, OllamaError
from .projects import ProjectItem

DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
CHUNK_CHARS = 600
TASK_CHARS = 1000  # how much of a task's body is used for matching

_STOPWORDS = set(
    """a an and are as at be by for from has have in into is it its of on or that the this to
    was were will with should can each all any we you they it's which when then than also
    not no do does done make use using via per page""".split()
)


@dataclass
class Match:
    item: ProjectItem
    score: float  # 0..1, higher is more similar


Log = Callable[[str], None]


class SimilarityFinder:
    def __init__(
        self,
        embed_model: Optional[str] = None,
        ollama: Optional[OllamaClient] = None,
        use_embeddings: bool = True,
        log: Log = lambda msg: None,
    ):
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._ollama = ollama or OllamaClient()
        self._use_embeddings = use_embeddings
        self._log = log
        self.method = "embeddings" if use_embeddings else "keywords"

    def find(
        self, document: str, items: Sequence[ProjectItem], top_k: int = 15, min_score: float = 0.0
    ) -> List[Match]:
        """Return up to `top_k` items most similar to `document`, best first."""
        candidates = [i for i in items if i.type != "REDACTED" and (i.title or "").strip()]
        chunks = chunk_text(document)
        if not candidates or not chunks:
            return []

        task_texts = [task_text(i) for i in candidates]
        scores = None
        if self._use_embeddings:
            try:
                scores = self._embedding_scores(chunks, task_texts)
                self.method = "embeddings"
            except OllamaError as exc:
                self._log(f"Embedding model unavailable ({exc}). Using keyword matching instead.")
        if scores is None:
            scores = _tfidf_scores(chunks, task_texts)
            self.method = "keywords"

        matches = [Match(item, round(s, 3)) for item, s in zip(candidates, scores) if s > min_score]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def _embedding_scores(self, chunks: List[str], tasks: List[str]) -> List[float]:
        vectors = self._ollama.embed(self.embed_model, chunks + tasks)
        chunk_vecs, task_vecs = vectors[: len(chunks)], vectors[len(chunks):]
        return [max(_cosine(t, c) for c in chunk_vecs) for t in task_vecs]


def task_text(item: ProjectItem) -> str:
    body = (item.body or "").strip()[:TASK_CHARS]
    return f"{item.title}\n{body}".strip()


def chunk_text(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split on blank lines / page markers, then pack paragraphs into ~`size`-char chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|^--- Page \d+ ---$", text, flags=re.M)]
    chunks: List[str] = []
    current = ""
    for para in filter(None, paragraphs):
        if len(para) > size:  # very long paragraph: flush what we have, then cut it up
            if current:
                chunks.append(current)
                current = ""
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size:]
        if current and len(current) + len(para) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


# -- keyword fallback -------------------------------------------------------


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9À-ɏ]+", text.lower())
    return [_stem(w) for w in words if len(w) > 2 and w not in _STOPWORDS]


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _tfidf_scores(chunks: List[str], tasks: List[str]) -> List[float]:
    docs = [Counter(_tokens(t)) for t in chunks + tasks]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def vec(counts: Counter) -> Dict[str, float]:
        return {w: (1 + math.log(c)) * idf[w] for w, c in counts.items()}

    vecs = [vec(d) for d in docs]
    chunk_vecs, task_vecs = vecs[: len(chunks)], vecs[len(chunks):]
    return [max((_sparse_cosine(t, c) for c in chunk_vecs), default=0.0) for t in task_vecs]


def _sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Score levels differ by method: embeddings score everything a bit similar, keywords don't.
_LEVELS = {"embeddings": (0.6, 0.45), "keywords": (0.3, 0.12)}


def strength(score: float, method: str) -> str:
    """'strong', 'possible' or 'weak' for a score from the given method."""
    strong, possible = _LEVELS.get(method, _LEVELS["keywords"])
    if score >= strong:
        return "strong"
    if score >= possible:
        return "possible"
    return "weak"
