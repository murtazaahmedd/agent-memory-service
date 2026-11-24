"""A LangGraph agent wired to the memory engine.

    recall ──► act ──► execute ──┬─ success ─────────────────────────────► learn ──► END
                ▲                └─ failure ─► reflect (distil a lesson) ─┐
                └──────────────────────── retry once ◄──────────────────────┘

The "act" step is a deterministic stand-in for an LLM: it starts from default tool parameters and applies any
``set key=value`` instruction found in the recalled memories. That keeps the benchmark reproducible and free,
while the memory engine it exercises is exactly what an LLM agent would use.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .engine import MemoryEngine
from .env import DEFAULT_PARAMS, Task, ToolEnvironment

SET_RE = re.compile(r"set (\w+)=(\w+)")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d\w*)\b")


class AgentState(TypedDict, total=False):
    task: Task
    recalled_ids: List[str]
    lessons: List[str]
    params: Dict[str, str]
    attempts: int
    success: bool
    first_try_success: bool
    error: str
    new_lesson: Optional[str]


def situation(task_text: str) -> str:
    """The reusable part of a task: drop names, dates and numbers (sentence-initial verbs are kept)."""
    text = task_text[:1].lower() + task_text[1:]
    text = ENTITY_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ?.").lower()


def distil(task_text: str, error: str) -> str:
    """Turn one failure into a reusable lesson: the situation, what went wrong, and the fix."""
    generic = situation(task_text)
    fix = SET_RE.search(error)
    return f"When asked to {generic}: {error.split('. Fix:')[0]}. Fix: set {fix.group(1)}={fix.group(2)}"


class MemoryAgent:
    def __init__(self, engine: Optional[MemoryEngine], name: str = "ops-agent", max_attempts: int = 2) -> None:
        self.engine = engine
        self.name = name
        self.max_attempts = max_attempts
        self.graph = self._build()

    # --------------------------------------------------------------- nodes
    def recall(self, state: AgentState) -> AgentState:
        if not self.engine:
            return {"recalled_ids": [], "lessons": [], "attempts": 0}
        hits = self.engine.recall(self.name, situation(state["task"].text), top_k=3)
        return {"recalled_ids": [h.memory.id for h in hits], "lessons": [h.memory.text for h in hits], "attempts": 0}

    def act(self, state: AgentState) -> AgentState:
        params = DEFAULT_PARAMS()
        for lesson in state.get("lessons", []):
            for key, value in SET_RE.findall(lesson):
                if key in params:
                    params[key] = value
        return {"params": params, "attempts": state.get("attempts", 0) + 1}

    def execute(self, state: AgentState) -> AgentState:
        ok, message = ToolEnvironment.execute(state["task"], state["params"])
        out: AgentState = {"success": ok, "error": "" if ok else message}
        if state["attempts"] == 1:
            out["first_try_success"] = ok
        return out

    def reflect(self, state: AgentState) -> AgentState:
        lesson = distil(state["task"].text, state["error"])
        return {"new_lesson": lesson, "lessons": state.get("lessons", []) + [lesson]}

    def learn(self, state: AgentState) -> AgentState:
        if not self.engine:
            return {}
        # memories that were in context get credit or blame for the final outcome
        if state.get("recalled_ids"):
            self.engine.feedback(state["recalled_ids"], state["success"])
        if state.get("new_lesson"):
            # a lesson learned from a failure, confirmed by the retry's outcome
            self.engine.write(self.name, "failure_lesson", state["new_lesson"], success=state["success"],
                              trigger=situation(state["task"].text))
        elif state["success"] and state.get("lessons"):
            for lesson in state["lessons"]:
                self.engine.write(self.name, "failure_lesson", lesson, success=True,
                                  trigger=situation(state["task"].text))
        return {}

    # --------------------------------------------------------------- graph
    def _after_execute(self, state: AgentState) -> str:
        if state["success"] or state["attempts"] >= self.max_attempts:
            return "learn"
        return "reflect"

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("recall", "act", "execute", "reflect", "learn"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("recall")
        g.add_edge("recall", "act")
        g.add_edge("act", "execute")
        g.add_conditional_edges("execute", self._after_execute, {"learn": "learn", "reflect": "reflect"})
        g.add_edge("reflect", "act")
        g.add_edge("learn", END)
        return g.compile()

    def run(self, task: Task) -> AgentState:
        return self.graph.invoke({"task": task})
