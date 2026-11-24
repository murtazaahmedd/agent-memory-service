"""A small tool-use environment with hidden rules, used to measure whether memory makes an agent better.

Each task type hides one rule (for example "the weather API only accepts ISO dates"). An agent that does not
know the rule fails, and the error message tells it what to change. An agent with memory can store that lesson
and get the next, differently worded task of the same type right on the first try.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass(frozen=True)
class TaskType:
    name: str
    templates: Tuple[str, ...]
    slots: Dict[str, Tuple[str, ...]]
    required: Tuple[str, str]          # (parameter, value) the hidden rule needs
    error: str                         # what the tool says when the rule is broken


CITIES = ("Karachi", "Lahore", "Dubai", "Istanbul", "Riyadh", "Doha")
PEOPLE = ("Ayesha", "Bilal", "Sara", "Hamza", "Zainab", "Omar")
CUSTOMERS = ("Acme Corp", "Globex", "Initech", "Umbrella", "Soylent")
DATES = ("next Monday", "the 14th", "tomorrow", "December 3rd", "this Friday")

TASK_TYPES: List[TaskType] = [
    TaskType(
        "weather",
        ("Get the weather forecast for {city} on {date}",
         "What will the weather be like in {city} {date}?",
         "Check the forecast in {city} for {date} before the site visit"),
        {"city": CITIES, "date": DATES},
        ("date_format", "iso"),
        "400 Bad Request: the weather API only accepts ISO 8601 dates. Fix: set date_format=iso",
    ),
    TaskType(
        "orders",
        ("How many orders did {customer} place last month?",
         "Count last month's orders for {customer}",
         "Give me the order count for {customer} for the previous month"),
        {"customer": CUSTOMERS},
        ("exclude_deleted", "true"),
        "Check failed: the order count included soft-deleted orders. Fix: set exclude_deleted=true",
    ),
    TaskType(
        "email",
        ("Email {person} the weekly revenue report",
         "Send the weekly revenue report to {person} by email",
         "Mail {person} this week's revenue report"),
        {"person": PEOPLE},
        ("subject_style", "short"),
        "Mail server rejected the message: subject lines over 60 characters are blocked. Fix: set subject_style=short",
    ),
    TaskType(
        "fx",
        ("Convert the {customer} invoice from USD to PKR",
         "What is the {customer} invoice worth in PKR?",
         "Turn the USD amount on the {customer} invoice into PKR"),
        {"customer": CUSTOMERS},
        ("rate_source", "central_bank"),
        "Finance check failed: invoices must use the central bank rate, not a market feed. Fix: set rate_source=central_bank",
    ),
]


@dataclass
class Task:
    type: TaskType
    text: str


class ToolEnvironment:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def sample(self) -> Task:
        tt = self.rng.choice(TASK_TYPES)
        template = self.rng.choice(tt.templates)
        values = {k: self.rng.choice(v) for k, v in tt.slots.items()}
        return Task(tt, template.format(**values))

    def stream(self, n: int) -> List[Task]:
        return [self.sample() for _ in range(n)]

    @staticmethod
    def execute(task: Task, params: Dict[str, str]) -> Tuple[bool, str]:
        key, value = task.type.required
        if params.get(key) == value:
            return True, "ok"
        return False, task.type.error


DEFAULT_PARAMS: Callable[[], Dict[str, str]] = lambda: {  # noqa: E731
    "date_format": "natural", "exclude_deleted": "false", "subject_style": "descriptive", "rate_source": "market",
}
