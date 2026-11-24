# Agent Memory Service

**Self-improving episodic memory for LLM agents.** An agent fails, writes down the lesson, and gets the next
similar task right the first time, even when the task is worded differently. Lessons that keep working are
promoted into skills; lessons that keep failing decay and stop being recalled.

```text
60 tasks x 5 seeds, 4 hidden rules, 3 phrasings each

                               no memory   with memory
first-try success, overall            0%           90%
  episodes 1-20                       0%           74%
  episodes 21-40                      0%           97%
  episodes 41-60                      0%          100%

memories kept by the novelty gate: 4.0 per run (from 60 episodes)
crystallized into skills:          4.0 per run
```

Sixty episodes collapse into four memories, one per real lesson, because the novelty gate merges repeats
instead of storing every event.

## How it works

```text
           ┌────────────── LangGraph agent ───────────────┐
 task ───► │ recall ─► act ─► execute ─┬─ ok ──────► learn │──► memory engine (SQLite)
           │             ▲             └─ fail ─► reflect │       write · recall · feedback
           │             └──────── retry ◄───────────┘     │
           └───────────────────────────────────────────────┘
```

**Write: the novelty gate.** A new lesson is compared with the agent's existing memories of the same kind. If
the cosine distance is below 0.35 it merges into the closest one (support + 1); otherwise it becomes a new memory.
Each memory keeps two vectors: the lesson itself (for the gate) and a running centroid of every situation it was
learned in (for recall), so one memory gradually learns many phrasings of the same problem.

**Usefulness: a Beta posterior.** Every memory carries `Beta(alpha, beta)` on "using this leads to success".
Outcomes of runs that had the memory in context update it, so `propensity = alpha / (alpha + beta)`.

**Lifecycle.**

| State | When |
|---|---|
| `ephemeral` | just written |
| `crystallized` | support >= 3 and propensity >= 0.6: a proven, reusable skill |
| `decayed` | at least 5 evaluations and propensity < 0.3: never recalled again |

**Recall: scored, with a hard floor.**

```text
score = 0.55 * relevance + 0.20 * freshness + 0.25 * usefulness
relevance  = cosine(situation, memory's situation centroid)   must be >= 0.25, or the memory is skipped
freshness  = 0.5 ^ (days since last seen / 7)
usefulness = 0.7 * propensity + 0.3 * [crystallized]
```

The floor matters: a fresh, highly useful memory about invoices can never be recalled for a question about the
weather.

**Reflection.** When a tool call fails, the agent distils a lesson: it strips names, dates and numbers from the
task, keeps the error and the fix ("When asked to get the weather forecast for ...: the API only accepts ISO 8601
dates. Fix: set date_format=iso").

## The benchmark

`memory_service/env.py` is a small tool-use world with four hidden rules (ISO dates for the weather API,
exclude soft-deleted orders, short email subjects, central-bank FX rates). Each rule has three different
phrasings and random names, cities and dates. An agent that doesn't know a rule fails, and the error tells it the
fix.

The agent's "act" step is a deterministic stand-in for an LLM: it applies any `set key=value` it finds in recalled
memories. That makes the benchmark free and reproducible, and isolates the thing being measured: the memory engine.
Swap in a real LLM call in `MemoryAgent.act` and nothing else changes.

## Quick start

```bash
pip install -r requirements.txt
make evals      # the benchmark above
make test       # 10 tests: merging, isolation, Beta updates, crystallization, decay, recall floor, API
make serve      # HTTP API on :8000, docs at /docs
```

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/events` | write a lesson `{agent, kind, text, trigger, success}`; returns created/updated and the distance |
| `GET` | `/recall?agent=&query=&top_k=` | ranked memories with relevance, freshness and usefulness |
| `POST` | `/feedback` | `{memory_ids, success}` after a run; returns any memories that decayed |
| `GET` | `/memories?agent=` | everything an agent remembers |

## Project layout

```text
memory_service/
  engine.py   MemoryEngine: novelty gate, Beta posterior, crystallize, decay, scored recall
  agent.py    LangGraph agent: recall -> act -> execute -> reflect/retry -> learn
  env.py      tool environment with hidden rules (the benchmark)
  api.py      FastAPI service
evals/        with-memory vs no-memory benchmark (run in CI)
tests/        unit and API tests
```

## Ideas to extend

- Replace the hashed bag-of-words embedder with BGE or OpenAI embeddings and store vectors in pgvector.
- Use an LLM for reflection, and let it decide when a lesson contradicts an existing one (supersede it).
- Per-memory credit assignment: only reward memories whose fix was actually applied.
- Add a semantic "facts" memory next to the episodic one, with different decay rules.
