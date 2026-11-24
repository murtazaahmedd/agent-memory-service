"""A self-improving episodic memory for agents.

Lifecycle of a memory:

    event ──► novelty gate ──► new memory            (nothing similar yet)
                        └────► update existing one   (cosine distance < threshold: same lesson, new evidence)
    every observation updates a Beta(alpha, beta) posterior on "this memory leads to success"
    support >= 3 and propensity >= 0.6 ──► crystallized (promoted into a reusable skill)
    feedback after a run that used the memory ──► Beta update; low propensity after enough runs ──► decayed

Recall ranks memories by relevance, freshness and usefulness, with a hard relevance floor so a fresh but
unrelated memory can never be recalled.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,              -- failure_lesson | success_pattern | fact
    text TEXT NOT NULL,              -- the lesson, in words
    embedding TEXT NOT NULL,         -- lesson text vector (used by the novelty gate)
    key_embedding TEXT NOT NULL,     -- situation vector: running centroid of every trigger seen (used by recall)
    support INTEGER NOT NULL,        -- how many events merged into it
    alpha REAL NOT NULL,             -- Beta posterior: successes + 1
    beta REAL NOT NULL,              -- Beta posterior: failures + 1
    eval_count INTEGER NOT NULL,     -- feedback events received
    state TEXT NOT NULL,             -- ephemeral | crystallized | decayed
    created_at REAL NOT NULL,
    last_observed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mem_agent_kind ON memories(agent, kind);
"""


@dataclass
class Config:
    novelty_threshold: float = 0.35     # cosine distance below this merges into an existing memory
    crystallize_support: int = 3
    crystallize_propensity: float = 0.6
    decay_propensity: float = 0.3
    decay_min_evals: int = 5
    min_relevance: float = 0.25         # hard floor on cosine similarity at recall
    weights: tuple = (0.55, 0.20, 0.25)  # relevance, freshness, usefulness
    half_life_days: float = 7.0


@dataclass
class Memory:
    id: str
    agent: str
    kind: str
    text: str
    support: int
    alpha: float
    beta: float
    eval_count: int
    state: str
    created_at: float
    last_observed_at: float
    embedding: np.ndarray = field(repr=False, default=None)
    key_embedding: np.ndarray = field(repr=False, default=None)

    @property
    def propensity(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class Recalled:
    memory: Memory
    score: float
    relevance: float
    freshness: float
    usefulness: float


@dataclass
class WriteResult:
    action: str          # created | updated
    memory_id: str
    distance: float
    crystallized: bool


class Embedder:
    """Stateless hashed bag of words + bigrams. Swap for BGE / OpenAI embeddings in production."""

    def __init__(self, dim: int = 2 ** 14) -> None:
        self._v = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False, norm="l2",
                                    stop_words="english")

    def __call__(self, text: str) -> np.ndarray:
        return self._v.transform([text]).toarray()[0]


class MemoryEngine:
    def __init__(self, path: str = ":memory:", config: Optional[Config] = None, embedder: Optional[Embedder] = None,
                 clock=time.time) -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.cfg = config or Config()
        self.embed = embedder or Embedder()
        self.clock = clock

    # ------------------------------------------------------------------ helpers
    def _row(self, r: sqlite3.Row) -> Memory:
        return Memory(r["id"], r["agent"], r["kind"], r["text"], r["support"], r["alpha"], r["beta"],
                      r["eval_count"], r["state"], r["created_at"], r["last_observed_at"],
                      np.array(json.loads(r["embedding"])), np.array(json.loads(r["key_embedding"])))

    def get(self, memory_id: str) -> Memory:
        return self._row(self.db.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone())

    def all(self, agent: Optional[str] = None) -> List[Memory]:
        q, args = "SELECT * FROM memories", []
        if agent:
            q, args = q + " WHERE agent = ?", [agent]
        return [self._row(r) for r in self.db.execute(q + " ORDER BY created_at", args)]

    def _save_embedding(self, vec: np.ndarray) -> str:
        return json.dumps([round(float(x), 6) for x in vec])

    # -------------------------------------------------------------------- write
    def write(self, agent: str, kind: str, text: str, success: bool, weight: float = 1.0,
              trigger: Optional[str] = None) -> WriteResult:
        """Admit an event. Similar lessons from the same agent and kind merge; new ones start a memory.

        ``text`` is the lesson; ``trigger`` is the situation it applies to (defaults to the lesson). The novelty
        gate compares lessons, while recall compares the new situation with every situation the memory has seen,
        so one memory learns many phrasings of the same problem.
        """
        vec = self.embed(text)
        key = self.embed(trigger or text)
        now = self.clock()
        candidates = [m for m in self.all(agent) if m.kind == kind and m.state != "decayed"]
        best, best_dist = None, 1.0
        for m in candidates:
            dist = 1.0 - float(vec @ m.embedding)
            if dist < best_dist:
                best, best_dist = m, dist

        with self.db:
            if best is not None and best_dist < self.cfg.novelty_threshold:
                # running centroid keeps the memory anchored on everything it has seen
                centroid = (best.key_embedding * best.support + key) / (best.support + 1)
                centroid /= np.linalg.norm(centroid) or 1.0
                self.db.execute(
                    "UPDATE memories SET support = support + 1, alpha = alpha + ?, beta = beta + ?, "
                    "key_embedding = ?, last_observed_at = ? WHERE id = ?",
                    [weight if success else 0.0, 0.0 if success else weight, self._save_embedding(centroid), now, best.id])
                memory_id, action = best.id, "updated"
            else:
                memory_id, action = uuid.uuid4().hex[:12], "created"
                self.db.execute(
                    "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, 0, 'ephemeral', ?, ?)",
                    [memory_id, agent, kind, text, self._save_embedding(vec), self._save_embedding(key),
                     1.0 + (weight if success else 0.0), 1.0 + (0.0 if success else weight), now, now])
                best_dist = 1.0 if best is None else best_dist
        return WriteResult(action, memory_id, round(best_dist, 4), self._maybe_crystallize(memory_id))

    def _maybe_crystallize(self, memory_id: str) -> bool:
        m = self.get(memory_id)
        if m.state == "ephemeral" and m.support >= self.cfg.crystallize_support \
                and m.propensity >= self.cfg.crystallize_propensity:
            with self.db:
                self.db.execute("UPDATE memories SET state = 'crystallized' WHERE id = ?", [memory_id])
            return True
        return m.state == "crystallized"

    # ------------------------------------------------------------------- recall
    def recall(self, agent: str, query: str, top_k: int = 3, kinds: Optional[List[str]] = None) -> List[Recalled]:
        vec, now = self.embed(query), self.clock()
        w_rel, w_fresh, w_use = self.cfg.weights
        out = []
        for m in self.all(agent):
            if m.state == "decayed" or (kinds and m.kind not in kinds):
                continue
            relevance = float(vec @ m.key_embedding)
            if relevance < self.cfg.min_relevance:
                continue   # hard floor: freshness and usefulness can never rescue an unrelated memory
            age_days = (now - m.last_observed_at) / 86400
            freshness = math.exp(-math.log(2) * age_days / self.cfg.half_life_days)
            usefulness = 0.7 * m.propensity + 0.3 * (1.0 if m.state == "crystallized" else 0.0)
            score = w_rel * relevance + w_fresh * freshness + w_use * usefulness
            out.append(Recalled(m, round(score, 4), round(relevance, 4), round(freshness, 4), round(usefulness, 4)))
        return sorted(out, key=lambda r: -r.score)[:top_k]

    # ----------------------------------------------------------------- feedback
    def feedback(self, memory_ids: List[str], success: bool) -> List[str]:
        """Credit or blame the memories that were in the prompt. Returns ids that decayed."""
        decayed = []
        with self.db:
            for mid in set(memory_ids):
                self.db.execute("UPDATE memories SET alpha = alpha + ?, beta = beta + ?, eval_count = eval_count + 1 "
                                "WHERE id = ?", [1.0 if success else 0.0, 0.0 if success else 1.0, mid])
                m = self.get(mid)
                if m.eval_count >= self.cfg.decay_min_evals and m.propensity < self.cfg.decay_propensity:
                    self.db.execute("UPDATE memories SET state = 'decayed' WHERE id = ?", [mid])
                    decayed.append(mid)
        return decayed

    def stats(self) -> dict:
        rows = self.db.execute("SELECT state, count(*) FROM memories GROUP BY state").fetchall()
        return {r[0]: r[1] for r in rows}
