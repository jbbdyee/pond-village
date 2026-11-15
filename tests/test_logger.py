"""logger.py 테스트: SPEC 7장 표 모양, 하루 기록, 연못 확인 2줄, 끝난 이유, 재현성, CLI 저장."""

import json
import sqlite3
from pathlib import Path

import pytest

from pondvillage import logger as lg
from pondvillage.agents.base import BaseAgent, Observation
from pondvillage.agents.scripted import CooperativeBot, GreedyBot
from pondvillage.runner import END_COLLAPSED, END_DAYS, main, run, run_baseline
from pondvillage.world import World, config_from_dict

BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

RAW = {
    "pond": {
        "capacity": 100, "initial_stock": 100, "regen_multiplier": 2,
        "collapse_below": 5, "estimate_noise": [0.8, 1.2],
    },
    "economy": {
        "start_money": 20, "fish_price": 1, "daily_cost": 5,
        "max_catch_per_day": 20, "inspect_catch_factor": 0.5, "bankrupt_after_days": 3,
    },
    "run": {"days": 30, "seed": 1},
    "agents": [{"id": f"a{i}", "name": f"어부{i}", "persona": "어부"} for i in range(1, 6)],
}
IDS = [f"a{i}" for i in range(1, 6)]

# SPEC 7장 "주요 컬럼"
SPEC_COLUMNS = {
    "runs": ["run_id", "config_name", "condition", "seed", "model", "started_at", "status"],
    "days": ["run_id", "day", "stock_start", "stock_end", "total_catch", "collapsed"],
    "agent_days": ["run_id", "day", "agent_id", "money", "alive", "catch", "violated", "beliefs_json", "trust_json"],
    "messages": ["run_id", "day", "from", "to", "text", "is_rumor"],
    "actions": ["run_id", "day", "phase", "agent_id", "action_json", "reason", "parse_error", "clamped"],
    "votes": ["run_id", "day", "proposal_id", "agent_id", "vote"],
    "llm_calls": [
        "run_id", "day", "agent_id", "phase", "model",
        "prompt_tokens", "completion_tokens", "latency_ms", "cost_usd",
        "prompt", "response", "parse_error",
    ],
}


def make_world() -> World:
    return World.from_config(config_from_dict(RAW))


class Scripted(BaseAgent):
    """정해 둔 답을 차례로 내는 가짜 어부."""

    def __init__(self, agent_id: str, answers: list) -> None:
        super().__init__(agent_id)
        self.answers = list(answers)

    def decide_fish(self, obs: Observation) -> dict:
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


@pytest.fixture
def conn(tmp_path):
    c = lg.open_db(tmp_path / "runs" / "test.db")
    yield c
    c.close()


def start(conn, run_id="r1", condition="coop", seed=1):
    return lg.RunLogger.start(
        conn, run_id, config_name="test", condition=condition, seed=seed, model="scripted",
        started_at="2026-11-14T00:00:00",
    )


def rows(conn, sql, *params):
    return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------- 표 모양


def test_open_db_creates_folder_and_spec_tables(tmp_path):
    path = tmp_path / "new" / "x.db"
    c = lg.open_db(path)
    assert path.exists()
    for table, cols in SPEC_COLUMNS.items():
        assert [r[1] for r in c.execute(f"PRAGMA table_info({table})")] == cols
    c.close()


def test_open_db_twice_keeps_existing_rows(tmp_path):
    path = tmp_path / "x.db"
    c = lg.open_db(path)
    start(c)
    c.close()
    c = lg.open_db(path)
    assert rows(c, "SELECT run_id FROM runs") == [("r1",)]
    c.close()


def test_run_id_and_stamp_shape():
    from datetime import datetime
    stamp = lg.make_stamp(datetime(2026, 11, 14, 15, 30, 0))
    assert stamp == "20261114-153000"
    assert lg.make_run_id(stamp, "coop", 3) == "20261114-153000_coop_s3"


# ---------------------------------------------------------------- 실행 기록


def test_start_marks_running(conn):
    start(conn)
    assert rows(conn, "SELECT config_name, condition, seed, model, status FROM runs") == [
        ("test", "coop", 1, "scripted", lg.STATUS_RUNNING)
    ]


def test_coop_run_rows_per_day_and_agent(conn):
    r = run(make_world(), {i: CooperativeBot(i) for i in IDS}, seed=1, days=3, logger=start(conn))
    assert rows(conn, "SELECT status FROM runs") == [(END_DAYS,)]
    assert rows(conn, "SELECT day, stock_start, stock_end, total_catch, collapsed FROM days ORDER BY day") == [
        (1, 100, 100, 50, 0), (2, 100, 100, 50, 0), (3, 100, 100, 50, 0)
    ]
    assert rows(conn, "SELECT COUNT(*) FROM agent_days") == [(15,)]
    assert rows(conn, "SELECT COUNT(*) FROM actions") == [(15,)]
    # 돈·잡은 수는 하루 결과와 같다
    last = r.days[-1]
    for agent_id, money, alive, catch, violated in rows(
        conn, "SELECT agent_id, money, alive, catch, violated FROM agent_days WHERE day = 3"
    ):
        assert (money, alive, catch, violated) == (last.money[agent_id], 1, 10, 0)


def test_action_row_keeps_amount_reason_and_flags(conn):
    agents = {i: CooperativeBot(i) for i in IDS}
    agents["a1"] = Scripted("a1", [{"action": "fish", "amount": 99, "reason": "많이"}])
    agents["a2"] = Scripted("a2", [{"action": "swim"}])
    run(make_world(), agents, seed=1, days=1, logger=start(conn))
    got = {a: (json.loads(j), reason, err, clamped) for a, j, reason, err, clamped in rows(
        conn, "SELECT agent_id, action_json, reason, parse_error, clamped FROM actions"
    )}
    assert got["a1"] == ({"action": "fish", "amount": 20}, "많이", None, 1)
    assert got["a2"][0] == {"action": "none", "amount": 0}
    assert got["a2"][2] is not None and got["a2"][3] == 0


def test_inspect_leaves_two_action_rows(conn):
    agents = {i: CooperativeBot(i) for i in IDS}
    agents["a1"] = Scripted("a1", [{"action": "inspect_pond"}, {"action": "fish", "amount": 8}])
    run(make_world(), agents, seed=1, days=1, logger=start(conn))
    got = rows(conn, "SELECT phase, action_json FROM actions WHERE agent_id = 'a1' ORDER BY rowid")
    assert [(p, json.loads(j)) for p, j in got] == [
        (lg.PHASE_FISH, {"action": "inspect_pond", "amount": 0}),
        (lg.PHASE_FISH_AFTER_INSPECT, {"action": "fish", "amount": 8}),
    ]
    assert rows(conn, "SELECT COUNT(*) FROM actions") == [(6,)]


def test_greedy_run_status_is_collapsed(conn):
    run(make_world(), {i: GreedyBot(i) for i in IDS}, seed=1, logger=start(conn, condition="greedy"))
    assert rows(conn, "SELECT status FROM runs") == [(END_COLLAPSED,)]
    assert rows(conn, "SELECT day, stock_end, collapsed FROM days") == [(1, 0, 1)]


def test_departed_fisher_is_still_logged(conn):
    run(make_world(), {i: CooperativeBot(i, amount=0) for i in IDS}, seed=1, logger=start(conn))
    assert rows(conn, "SELECT COUNT(*) FROM agent_days WHERE day = 7 AND alive = 0") == [(5,)]
    assert rows(conn, "SELECT COUNT(*) FROM actions WHERE day = 7") == [(5,)]


class Broken(BaseAgent):
    def decide_fish(self, obs: Observation) -> dict:
        if obs.day == 2:
            raise RuntimeError("고장")
        return {"action": "fish", "amount": 10}


def test_error_mid_run_marks_error_and_keeps_earlier_days(conn):
    agents = {i: CooperativeBot(i) for i in IDS}
    agents["a1"] = Broken("a1")
    with pytest.raises(RuntimeError):
        run(make_world(), agents, seed=1, logger=start(conn))
    assert rows(conn, "SELECT status FROM runs") == [(lg.STATUS_ERROR,)]
    assert rows(conn, "SELECT day FROM days") == [(1,)]


def test_unused_tables_stay_empty(conn):
    run(make_world(), {i: CooperativeBot(i) for i in IDS}, seed=1, days=2, logger=start(conn))
    for table in ("messages", "votes", "llm_calls"):
        assert rows(conn, f"SELECT COUNT(*) FROM {table}") == [(0,)]


# ---------------------------------------------------------------- 재현성 · 기준선


def _dump(c):
    return [
        rows(c, "SELECT * FROM days ORDER BY day"),
        rows(c, "SELECT * FROM agent_days ORDER BY day, agent_id"),
        rows(c, "SELECT * FROM actions ORDER BY rowid"),
    ]


def test_same_seed_same_records(tmp_path):
    from pondvillage.runner import build_bots
    dumps = []
    for name in ("a.db", "b.db"):
        c = lg.open_db(tmp_path / name)
        world = make_world()
        run(world, build_bots(world.config, "random", 5), seed=5, days=10, logger=start(c, condition="random", seed=5))
        dumps.append(_dump(c))
        c.close()
    assert dumps[0] == dumps[1]


def test_baseline_puts_all_runs_in_one_file(conn):
    summaries = run_baseline(make_world().config, [1, 2], conn=conn, config_name="test", stamp="S")
    got = rows(conn, "SELECT run_id, condition, seed, status FROM runs ORDER BY rowid")
    assert [g[0] for g in got] == ["S_greedy_s1", "S_greedy_s2", "S_coop_s1", "S_coop_s2", "S_random_s1", "S_random_s2"]
    assert [g[3] for g in got] == [s.end_reason for s in summaries]
    # 요약의 일수는 days 표 줄 수와 같다
    for (run_id, *_), s in zip(got, summaries):
        assert rows(conn, "SELECT COUNT(*) FROM days WHERE run_id = ?", run_id) == [(s.days,)]


# ---------------------------------------------------------------- CLI


def test_cli_saves_db_and_viewer_shows_it(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    assert main(["--config", str(BASE_YAML), "--days", "2", "--bot", "coop", "--runs-dir", str(runs_dir)]) == 0
    files = list(runs_dir.glob("*_coop_s1.db"))
    assert len(files) == 1
    capsys.readouterr()
    assert lg.main([str(files[0])]) == 0
    out = capsys.readouterr().out
    assert "days" in out and "agent_01" in out and "170" not in out  # 2일만 진행
    assert "30" in out  # 2일째 돈 20 + 2 × 5


def test_cli_no_log_makes_no_file(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    main(["--config", str(BASE_YAML), "--days", "1", "--bot", "coop", "--runs-dir", str(runs_dir), "--no-log"])
    assert not runs_dir.exists()


def test_cli_baseline_saves_one_file(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    main(["--config", str(BASE_YAML), "--baseline", "--seeds", "2", "--runs-dir", str(runs_dir)])
    files = list(runs_dir.glob("*_baseline.db"))
    assert len(files) == 1
    c = sqlite3.connect(files[0])
    assert rows(c, "SELECT COUNT(*) FROM runs") == [(6,)]
    assert rows(c, "SELECT DISTINCT config_name FROM runs") == [("base",)]
    c.close()


def test_viewer_missing_file(tmp_path, capsys):
    assert lg.main([str(tmp_path / "없음.db")]) == 1


# ---------------------------------------------------------------- LLM 호출 기록 (P4)


def test_llm_call_is_saved_with_prompt_response_and_parse_error(conn):
    from pondvillage.llm import CallRecord

    logger = start(conn)
    logger.log_llm_call(CallRecord(
        day=2, agent_id="a1", phase="fish", model="qwen2.5:7b", prompt='[{"role": "user"}]',
        response="물고기", prompt_tokens=100, completion_tokens=5, latency_ms=800, cost_usd=0.0,
        parse_error="JSON 아님",
    ))
    logger.finish(lg.STATUS_ABORTED_COST)
    assert rows(conn, "SELECT day, agent_id, phase, model, response, parse_error, cost_usd FROM llm_calls") == [
        (2, "a1", "fish", "qwen2.5:7b", "물고기", "JSON 아님", 0.0)
    ]
    assert rows(conn, "SELECT status FROM runs") == [("aborted_cost",)]


# ---------------------------------------------------------------- 질문지 길이 표 (P5)


def _call(day, phase, tokens, retry=False):
    from pondvillage.llm import CallRecord

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    if retry:
        msgs += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "다시"}]
    return CallRecord(
        day=day, agent_id="a1", phase=phase, model="m", prompt=json.dumps(msgs, ensure_ascii=False),
        response="{}", prompt_tokens=tokens, completion_tokens=1, latency_ms=1, cost_usd=0.0, parse_error=None,
    )


def test_token_rows_skip_retries_and_table_shows_avg_and_max(conn):
    logger = start(conn)
    for rec in [_call(1, "fish", 400), _call(1, "fish", 500), _call(1, "fish", 900, retry=True),
                _call(1, "diary", 600), _call(2, "fish", 700)]:
        logger.log_llm_call(rec)
    logger.finish("days")
    got = lg.token_rows(conn, "r1")
    assert got == [(1, "fish", 400), (1, "fish", 500), (1, "diary", 600), (2, "fish", 700)]
    table = lg.format_tokens(got)
    assert "어획 평균" in table and "일기 최대" in table
    day1, day2 = table.splitlines()[1:3]
    assert day1.split() == ["1", "450", "500", "600", "600"]
    assert day2.split() == ["2", "700", "700", "-", "-"]


def test_cli_tokens_option(tmp_path, capsys):
    db = tmp_path / "t.db"
    c = lg.open_db(db)
    logger = lg.RunLogger.start(c, "r1", config_name="t", condition="llm", seed=1, model="m")
    logger.log_llm_call(_call(1, "fish", 455))
    logger.finish("days")
    c.close()
    assert lg.main([str(db), "--tokens"]) == 0
    out = capsys.readouterr().out
    assert "질문지 길이" in out and "455" in out
