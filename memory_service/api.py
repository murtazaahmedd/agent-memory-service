"""HTTP API for the memory engine, so any agent (Python, TypeScript, n8n...) can use it."""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .engine import MemoryEngine

engine = MemoryEngine(os.getenv("MEMORY_DB", "memory.sqlite"))
app = FastAPI(title="Agent Memory Service", version="0.1.0")


class EventIn(BaseModel):
    agent: str
    kind: str = Field("failure_lesson", examples=["failure_lesson", "success_pattern", "fact"])
    text: str = Field(..., description="the lesson, in words")
    trigger: Optional[str] = Field(None, description="the situation it applies to")
    success: bool


class FeedbackIn(BaseModel):
    memory_ids: List[str]
    success: bool


@app.post("/events")
def write_event(event: EventIn):
    r = engine.write(event.agent, event.kind, event.text, event.success, trigger=event.trigger)
    return r.__dict__


@app.get("/recall")
def recall(agent: str, query: str, top_k: int = 3):
    return [{"id": h.memory.id, "text": h.memory.text, "state": h.memory.state, "score": h.score,
             "relevance": h.relevance, "freshness": h.freshness, "usefulness": h.usefulness}
            for h in engine.recall(agent, query, top_k)]


@app.post("/feedback")
def feedback(body: FeedbackIn):
    return {"decayed": engine.feedback(body.memory_ids, body.success)}


@app.get("/memories")
def memories(agent: Optional[str] = None):
    return [{"id": m.id, "kind": m.kind, "text": m.text, "support": m.support, "state": m.state,
             "propensity": round(m.propensity, 3)} for m in engine.all(agent)]


@app.get("/health")
def health():
    return {"status": "ok", "memories": engine.stats()}
