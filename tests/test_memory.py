from fastapi.testclient import TestClient

from memory_service.agent import MemoryAgent, distil, situation
from memory_service.engine import Config, MemoryEngine
from memory_service.env import ToolEnvironment

LESSON = "When calling the weather API: it only accepts ISO dates. Fix: set date_format=iso"


def test_similar_lessons_merge_and_different_ones_do_not():
    eng = MemoryEngine()
    a = eng.write("a", "failure_lesson", LESSON, success=False)
    b = eng.write("a", "failure_lesson", LESSON + " again", success=True)
    c = eng.write("a", "failure_lesson", "Invoices must use the central bank rate. Fix: set rate_source=central_bank", success=False)
    assert (a.action, b.action, c.action) == ("created", "updated", "created")
    assert a.memory_id == b.memory_id != c.memory_id
    assert eng.get(a.memory_id).support == 2


def test_memories_are_isolated_per_agent_and_kind():
    eng = MemoryEngine()
    a = eng.write("agent-a", "failure_lesson", LESSON, success=True)
    b = eng.write("agent-b", "failure_lesson", LESSON, success=True)
    c = eng.write("agent-a", "fact", LESSON, success=True)
    assert len({a.memory_id, b.memory_id, c.memory_id}) == 3
    assert eng.recall("agent-b", "weather API dates")[0].memory.id == b.memory_id


def test_beta_posterior_tracks_outcomes():
    eng = MemoryEngine()
    mid = eng.write("a", "failure_lesson", LESSON, success=False).memory_id
    assert eng.get(mid).propensity == 1 / 3          # Beta(1, 2)
    eng.write("a", "failure_lesson", LESSON, success=True)
    assert eng.get(mid).propensity == 2 / 4          # Beta(2, 2)


def test_crystallizes_after_enough_successful_support():
    eng = MemoryEngine()
    results = [eng.write("a", "failure_lesson", LESSON, success=True) for _ in range(3)]
    assert [r.crystallized for r in results] == [False, False, True]
    assert eng.get(results[0].memory_id).state == "crystallized"


def test_bad_memory_decays_after_repeated_failures_and_is_not_recalled():
    eng = MemoryEngine(config=Config(decay_min_evals=5))
    mid = eng.write("a", "failure_lesson", LESSON, success=False).memory_id
    decayed = []
    for _ in range(5):
        decayed += eng.feedback([mid], success=False)
    assert decayed == [mid]
    assert eng.recall("a", "weather API ISO dates") == []


def test_relevance_floor_blocks_unrelated_memories():
    eng = MemoryEngine()
    eng.write("a", "failure_lesson", LESSON, success=True)
    assert eng.recall("a", "bake sourdough bread at home") == []


def test_freshness_decays_with_age():
    now = [1_000_000.0]
    eng = MemoryEngine(clock=lambda: now[0])
    eng.write("a", "failure_lesson", LESSON, success=True)
    fresh = eng.recall("a", "weather API ISO dates")[0].freshness
    now[0] += 7 * 86400   # one half-life later
    assert abs(eng.recall("a", "weather API ISO dates")[0].freshness - fresh / 2) < 1e-3


def test_distil_removes_entities_and_keeps_fix():
    lesson = distil("Get the weather forecast for Karachi on December 3rd",
                    "400 Bad Request: ISO only. Fix: set date_format=iso")
    assert "Karachi" not in lesson and "set date_format=iso" in lesson
    assert situation("Email Sara the weekly revenue report") == "email the weekly revenue report"


def test_agent_with_memory_beats_agent_without():
    tasks = ToolEnvironment(3).stream(40)
    baseline, agent = MemoryAgent(None), MemoryAgent(MemoryEngine())
    without = sum(baseline.run(t)["first_try_success"] for t in tasks)
    with_mem = sum(agent.run(t)["first_try_success"] for t in tasks)
    assert without == 0 and with_mem >= 30


def test_api_round_trip(tmp_path, monkeypatch):
    from memory_service import api
    monkeypatch.setattr(api, "engine", MemoryEngine(str(tmp_path / "m.sqlite")))
    client = TestClient(api.app)
    mid = client.post("/events", json={"agent": "a", "text": LESSON, "success": True}).json()["memory_id"]
    hits = client.get("/recall", params={"agent": "a", "query": "weather API ISO dates"}).json()
    assert hits[0]["id"] == mid
    assert client.post("/feedback", json={"memory_ids": [mid], "success": True}).json() == {"decayed": []}
    assert client.get("/health").json()["status"] == "ok"
