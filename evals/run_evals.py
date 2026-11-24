"""Benchmark: the same stream of tasks, run by an agent with and without memory.

Reports first-try success rate, success over time, how many memories the novelty gate kept, and how many
crystallized. Exits non-zero if memory does not beat the baseline by the required margin (used in CI).
"""
from __future__ import annotations

import sys

from memory_service.agent import MemoryAgent
from memory_service.engine import MemoryEngine
from memory_service.env import TASK_TYPES, ToolEnvironment

EPISODES, SEEDS, WINDOW = 60, (1, 2, 3, 4, 5), 20


def run_one(seed: int, with_memory: bool):
    engine = MemoryEngine() if with_memory else None
    agent = MemoryAgent(engine)
    results = [agent.run(task)["first_try_success"] for task in ToolEnvironment(seed).stream(EPISODES)]
    return results, engine


def main(min_gain: float = 0.5) -> int:
    base, mem, kept, crystal = [], [], [], []
    for seed in SEEDS:
        b, _ = run_one(seed, False)
        m, engine = run_one(seed, True)
        base += [b]
        mem += [m]
        kept.append(len(engine.all()))
        crystal.append(engine.stats().get("crystallized", 0))

    def rate(runs, lo=0, hi=EPISODES):
        vals = [x for r in runs for x in r[lo:hi]]
        return sum(vals) / len(vals)

    print(f"{EPISODES} tasks x {len(SEEDS)} seeds, {len(TASK_TYPES)} hidden rules, 3 phrasings each\n")
    print(f"{'':28}{'no memory':>12}{'with memory':>14}")
    print(f"{'first-try success, overall':28}{rate(base):>12.0%}{rate(mem):>14.0%}")
    for lo in range(0, EPISODES, WINDOW):
        label = f"  episodes {lo + 1}-{lo + WINDOW}"
        print(f"{label:28}{rate(base, lo, lo + WINDOW):>12.0%}{rate(mem, lo, lo + WINDOW):>14.0%}")
    print(f"\nmemories kept by the novelty gate: {sum(kept) / len(kept):.1f} per run (from {EPISODES} episodes)")
    print(f"crystallized into skills:          {sum(crystal) / len(crystal):.1f} per run")
    gain = rate(mem) - rate(base)
    return 0 if gain >= min_gain else 1


if __name__ == "__main__":
    sys.exit(main())
