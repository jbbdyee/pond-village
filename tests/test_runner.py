"""runner.py 테스트: 아침 추정치, 하루 진행 순서, 연못 확인, 종료 조건, 콘솔 표."""

import copy
import random
from pathlib import Path

import pytest

from pondvillage.actions import FISH, NONE
from pondvillage.agents.base import BaseAgent, Observation
from pondvillage.agents.scripted import CooperativeBot, GreedyBot
from pondvillage.runner import (
    END_ALL_LEFT,
    END_COLLAPSED,
    END_DAYS,
    build_bots,
    main,
    make_rng,
    morning_estimate,
    run,
    run_baseline,
    run_day,
    summarize,
)
from pondvillage.world import World, config_from_dict, load_config

BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

RAW = {
    "pond": {
        "capacity": 100,
        "initial_stock": 100,
        "regen_multiplier": 2,
        "collapse_below": 5,
        "estimate_noise": [0.8, 1.2],
    },
    "economy": {
        "start_money": 20,
        "fish_price": 1,
        "daily_cost": 5,
        "max_catch_per_day": 20,
        "inspect_catch_factor": 0.5,
        "bankrupt_after_days": 3,
    },
    "run": {"days": 30, "seed": 1},
    "agents": [{"id": f"a{i}", "name": f"어부{i}", "persona": "어부"} for i in range(1, 6)],
}
IDS = [f"a{i}" for i in range(1, 6)]


def make_world(**changes) -> World:
    raw = copy.deepcopy(RAW)
    for section, values in changes.items():
        raw[section].update(values)
    return World.from_config(config_from_dict(raw))


def all_bots(make) -> dict[str, BaseAgent]:
    return {i: make(i) for i in IDS}


class Scripted(BaseAgent):
    """정해 둔 답을 차례로 내고, 받은 관찰을 기록하는 가짜 어부."""

    def __init__(self, agent_id: str, answers: list) -> None:
        super().__init__(agent_id)
        self.answers = list(answers)
        self.seen: list[Observation] = []

    def decide_fish(self, obs: Observation) -> dict:
        self.seen.append(obs)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


# ---------------------------------------------------------------- 아침 추정치


def test_estimate_stays_within_noise_range():
    rng = random.Random(0)
    for stock in [0, 1, 5, 37, 100]:
        for _ in range(200):
            e = morning_estimate(stock, (0.8, 1.2), rng)
            assert round(stock * 0.8) <= e <= round(stock * 1.2)


def test_estimate_is_same_for_same_seed():
    def five_days(seed):
        rng = make_rng(seed, "estimate")
        return [morning_estimate(100, (0.8, 1.2), rng) for _ in range(5)]

    assert five_days(7) == five_days(7)
    assert five_days(7) != five_days(8)


def test_everyone_sees_same_estimate_and_no_exact_stock():
    world = make_world()
    agents = {i: Scripted(i, [{"action": "fish", "amount": 10}]) for i in IDS}
    result = run_day(world, agents, make_rng(1, "estimate"))
    seen = [agents[i].seen[0] for i in IDS]
    assert {o.stock_estimate for o in seen} == {result.estimate}
    assert all(o.exact_stock is None for o in seen)
    assert all(o.yesterday_total_catch is None for o in seen)  # 1일차


def test_yesterday_total_is_announced_next_morning():
    world = make_world()
    agents = {i: Scripted(i, [{"action": "fish", "amount": 7}]) for i in IDS}
    run(world, agents, seed=1, days=2)
    assert agents["a1"].seen[1].yesterday_total_catch == 35
    assert agents["a1"].seen[1].day == 2


# ---------------------------------------------------------------- 하루 정산


def test_coop_day_settles_in_spec_order():
    world = make_world()
    d = run_day(world, all_bots(CooperativeBot), make_rng(1, "estimate"))
    assert d.catches == {i: 10 for i in IDS}
    assert (d.stock_start, d.stock_after_catch, d.stock_end) == (100, 50, 100)
    assert d.money == {i: 25 for i in IDS}  # 20 + 10 - 5
    assert not d.collapsed and d.left == []


def test_over_requests_are_split_by_ratio():
    world = make_world(pond={"initial_stock": 50})
    d = run_day(world, all_bots(GreedyBot), make_rng(1, "estimate"))
    assert d.requests == {i: 20 for i in IDS}
    assert d.catches == {i: 10 for i in IDS}  # 50마리를 100마리 요청 비율로


def test_clamped_request_is_capped_at_20():
    world = make_world()
    agents = all_bots(CooperativeBot)
    agents["a1"] = Scripted("a1", [{"action": "fish", "amount": 99}])
    d = run_day(world, agents, make_rng(1, "estimate"))
    turn = next(t for t in d.turns if t.agent_id == "a1")
    assert turn.decision.clamped and d.requests["a1"] == 20


def test_bad_answer_does_nothing_but_still_pays_living_cost():
    world = make_world()
    agents = all_bots(CooperativeBot)
    agents["a1"] = Scripted("a1", [{"action": "dance"}])
    d = run_day(world, agents, make_rng(1, "estimate"))
    turn = next(t for t in d.turns if t.agent_id == "a1")
    assert turn.decision.kind == NONE and turn.decision.parse_error
    assert d.catches["a1"] == 0 and d.money["a1"] == 15


def test_departed_fisher_is_not_asked():
    world = make_world()
    world.get_agent("a1").alive = False
    agents = {i: Scripted(i, [{"action": "fish", "amount": 10}]) for i in IDS}
    d = run_day(world, agents, make_rng(1, "estimate"))
    assert agents["a1"].seen == []
    assert "a1" not in d.requests


# ---------------------------------------------------------------- 연못 확인 (inspect_pond)


def test_inspect_shows_exact_stock_and_halves_cap():
    world = make_world(pond={"initial_stock": 80})
    agents = all_bots(CooperativeBot)
    agents["a1"] = Scripted("a1", [{"action": "inspect_pond"}, {"action": "fish", "amount": 20}])
    d = run_day(world, agents, make_rng(1, "estimate"))
    first, second = agents["a1"].seen
    assert first.exact_stock is None
    assert second.exact_stock == 80 and second.stock_estimate == first.stock_estimate
    turn = next(t for t in d.turns if t.agent_id == "a1")
    assert turn.inspected and turn.first.kind == "inspect_pond"
    assert turn.decision.kind == FISH and turn.decision.amount == 10 and turn.decision.clamped


def test_inspect_twice_means_zero_catch():
    world = make_world()
    agents = all_bots(CooperativeBot)
    agents["a1"] = Scripted("a1", [{"action": "inspect_pond"}])
    d = run_day(world, agents, make_rng(1, "estimate"))
    turn = next(t for t in d.turns if t.agent_id == "a1")
    assert turn.inspected and turn.decision.kind == NONE and turn.decision.parse_error is None
    assert d.catches["a1"] == 0
    assert len(agents["a1"].seen) == 2  # 세 번은 묻지 않는다


# ---------------------------------------------------------------- 종료 조건


def test_greedy_village_collapses_on_day_one():
    world = make_world()
    r = run(world, all_bots(GreedyBot), seed=1)
    assert r.end_reason == END_COLLAPSED and len(r.days) == 1
    assert r.days[0].stock_end == 0 and world.pond.collapsed


def test_coop_village_lasts_30_days():
    world = make_world()
    r = run(world, all_bots(CooperativeBot), seed=1)
    assert r.end_reason == END_DAYS and len(r.days) == 30
    assert all(d.stock_end == 100 for d in r.days)
    assert r.days[-1].money == {i: 170 for i in IDS}  # 20 + 30 × 5
    assert world.day == 30


def test_village_that_never_fishes_empties_on_day_seven():
    world = make_world()
    r = run(world, all_bots(lambda i: CooperativeBot(i, amount=0)), seed=1)
    assert r.end_reason == END_ALL_LEFT and len(r.days) == 7
    assert sorted(r.days[-1].left) == IDS


def test_days_argument_overrides_config():
    r = run(make_world(), all_bots(CooperativeBot), seed=1, days=3)
    assert r.end_reason == END_DAYS and [d.day for d in r.days] == [1, 2, 3]


# ---------------------------------------------------------------- 기준선 (시드 1~10)

SEEDS = list(range(1, 11))


def test_baseline_greedy_collapses_on_every_seed():
    rows = run_baseline(make_world().config, SEEDS, kinds=("greedy",))
    assert all(r.end_reason == END_COLLAPSED and r.days == 1 and r.stock_end == 0 for r in rows)


def test_baseline_coop_lasts_30_days_on_every_seed():
    rows = run_baseline(make_world().config, SEEDS, kinds=("coop",))
    assert all(r.end_reason == END_DAYS and r.days == 30 and r.stock_end == 100 for r in rows)
    assert all(r.left == 0 and r.avg_money == 170 and r.total_catch == 1500 for r in rows)


def test_baseline_random_is_reproducible():
    config = make_world().config
    assert run_baseline(config, SEEDS, kinds=("random",)) == run_baseline(config, SEEDS, kinds=("random",))


def test_summarize_counts_departed_and_money():
    world = make_world()
    r = run(world, all_bots(lambda i: CooperativeBot(i, amount=0)), seed=1)
    s = summarize("idle", 1, r)
    assert s.left == 5 and s.days == 7 and s.total_catch == 0
    assert s.avg_money == r.days[-1].money["a1"]


# ---------------------------------------------------------------- 재현성


def _summary(seed: int):
    world = make_world()
    r = run(world, build_bots(world.config, "random", seed), seed=seed, days=10)
    return [(d.estimate, d.requests, d.catches, d.stock_end) for d in r.days]


def test_same_seed_gives_same_run():
    assert _summary(3) == _summary(3)


def test_different_seed_gives_different_run():
    assert _summary(3) != _summary(4)


def test_random_bots_get_separate_rngs():
    bots = build_bots(make_world().config, "random", seed=1)
    assert len({id(b.rng) for b in bots.values()}) == 5


def test_unknown_bot_kind_stops():
    with pytest.raises(ValueError):
        build_bots(make_world().config, "lazy", seed=1)


# ---------------------------------------------------------------- 콘솔 표


def test_cli_prints_table_with_names(capsys):
    assert main(["--config", str(BASE_YAML), "--days", "2", "--bot", "coop", "--no-log"]) == 0
    out = capsys.readouterr().out
    for name in [a.name for a in load_config(BASE_YAML).agents]:
        assert name in out
    assert "정한 날짜까지 버팀 (2일 진행)" in out


def test_cli_baseline_prints_summary(capsys):
    assert main(["--config", str(BASE_YAML), "--baseline", "--seeds", "2", "--no-log"]) == 0
    out = capsys.readouterr().out
    assert "탐욕: 2번 중 연못 붕괴 2" in out
    assert "협력: 2번 중 정한 날짜까지 버팀 2" in out
    assert "랜덤: 2번 중" in out


def test_cli_greedy_reports_collapse(capsys):
    main(["--config", str(BASE_YAML), "--bot", "greedy", "--no-log"])
    out = capsys.readouterr().out
    assert "붕괴" in out and "(1일 진행)" in out


# ---------------------------------------------------------------- LLM 마을 (P4, 진짜 LLM 없이)

import sqlite3
from types import SimpleNamespace


from openai import APIConnectionError

from pondvillage import llm
from pondvillage.agents.llm_agent import build_llm_agents
from pondvillage.logger import RunLogger, open_db
from pondvillage.runner import assign_models, village_condition, village_label


class FakeOllama:
    """openai 클라이언트 흉내. 답 목록을 돌려 쓰고, 모델 목록을 알려 준다."""

    def __init__(self, answers, models=("qwen2.5:7b", "gemma3:4b"), diary=None):
        self.answers = list(answers)
        self.diary = diary  # 주면 일기 질문에는 이 답을 준다. 없으면 일기에도 answers를 돌려 쓴다
        self.prompt_chars = False  # True면 prompt_tokens 대신 보낸 글자 수를 돌려준다 (길이 확인용)
        self.i = 0
        self.asked_models = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in models])

    def _create(self, **kw):
        self.asked_models.append(kw["model"])
        is_diary = "일기를 쓴다" in kw["messages"][0]["content"]
        if is_diary and self.diary is not None:
            answer = self.diary
        else:
            answer = self.answers[self.i % len(self.answers)]
            self.i += 1
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(
                prompt_tokens=sum(len(m["content"]) for m in kw["messages"]) if self.prompt_chars else 300,
                completion_tokens=20,
            ),
        )


FISH10 = '{"action": "fish", "amount": 10, "reason": "나눠 쓰자"}'


def llm_config():
    raw = copy.deepcopy(RAW)
    raw["llm"] = {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "cost_cap_usd": 1.0}
    return config_from_dict(raw)


def test_assign_models_alternates_override():
    config = llm_config()
    assert list(assign_models(config, ["q", "g"]).values()) == ["q", "g", "q", "g", "q"]
    assert set(assign_models(config).values()) == {"qwen2.5:7b"}


def test_village_label_and_condition():
    same = {i: "qwen2.5:7b" for i in IDS}
    mixed = dict(zip(IDS, ["q", "g", "q", "g", "q"]))
    assert (village_label(same), village_condition(same)) == ("qwen2.5:7b", "llm-same")
    assert (village_label(mixed), village_condition(mixed)) == ("3q+2g", "llm-mixed")


def _llm_run(tmp_path, answers, models, days=5, cap=1.0, diary=None):
    config = llm_config()
    world = World.from_config(config)
    conn = open_db(tmp_path / "llm.db")
    logger = RunLogger.start(conn, "r", config_name="t", condition="llm", seed=1, model=village_label(models))
    client = FakeOllama(answers, diary=diary)
    caller = llm.LLMCaller(client=client, meter=llm.CostMeter(cap), record=logger.log_llm_call)
    agents = build_llm_agents(config, caller, models)
    return conn, client, lambda: run(world, agents, seed=1, days=days, logger=logger)


def test_mixed_llm_village_runs_five_days_and_logs_calls(tmp_path):
    models = dict(zip(IDS, ["qwen2.5:7b", "gemma3:4b"] * 3))
    conn, client, go = _llm_run(tmp_path, [FISH10], models)
    result = go()
    assert result.end_reason == END_DAYS and len(result.days) == 5
    assert all(d.total_catch == 50 for d in result.days)
    assert client.asked_models[:5] == list(models.values())
    calls = conn.execute("SELECT COUNT(*), SUM(parse_error IS NULL), SUM(cost_usd) FROM llm_calls").fetchone()
    assert calls == (50, 50, 0.0)  # 하루 어획 5 + 일기 5
    conn.close()


def test_bad_answers_do_nothing_and_never_crash(tmp_path):
    models = {i: "qwen2.5:7b" for i in IDS}
    conn, _, go = _llm_run(tmp_path, ["몰라요"], models, days=2)
    result = go()
    assert len(result.days) == 2
    assert all(d.total_catch == 0 for d in result.days)
    errors = conn.execute(
        "SELECT phase, COUNT(*) FROM actions WHERE parse_error LIKE '두 번 모두%' GROUP BY phase ORDER BY phase"
    ).fetchall()
    assert errors == [("diary", 10), ("fish", 10)]
    assert conn.execute("SELECT COUNT(*) FROM llm_calls WHERE parse_error IS NOT NULL").fetchone() == (40,)
    assert conn.execute("SELECT status FROM runs").fetchone() == ("days",)
    conn.close()


def test_server_dying_mid_run_stops_cleanly_with_error_status(tmp_path):
    down = APIConnectionError(request=None)
    models = {i: "qwen2.5:7b" for i in IDS}
    conn, _, go = _llm_run(tmp_path, [FISH10] * 10 + [down] * 2, models)  # 1일째 어획·일기 뒤 꺼짐
    with pytest.raises(llm.LLMUnavailable):
        go()
    assert conn.execute("SELECT status FROM runs").fetchone() == ("error",)
    assert conn.execute("SELECT COUNT(*) FROM days").fetchone() == (1,)
    conn.close()


def test_cost_cap_aborts_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "estimate_cost", lambda *a: 0.3)
    models = {i: "qwen2.5:7b" for i in IDS}
    conn, _, go = _llm_run(tmp_path, [FISH10], models, cap=1.0)
    with pytest.raises(llm.CostCapExceeded):
        go()
    assert conn.execute("SELECT status FROM runs").fetchone() == ("aborted_cost",)
    # 상한을 넘긴 4번째 호출까지 기록된다
    assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone() == (4,)
    conn.close()


def test_main_llm_stops_early_when_ollama_is_down(tmp_path, monkeypatch, capsys):
    def down(*a, **kw):
        raise llm.LLMUnavailable("WSL을 깨워 주세요")

    monkeypatch.setattr(llm, "check_ollama", down)
    code = main(["--config", str(BASE_YAML), "--llm", "--days", "1", "--runs-dir", str(tmp_path)])
    assert code == 1
    assert "WSL" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # 기록 파일도 만들지 않는다


def test_main_llm_mixed_village_end_to_end(tmp_path, monkeypatch, capsys):
    fake = FakeOllama([FISH10])
    monkeypatch.setattr(llm, "make_client", lambda *a, **kw: fake)
    code = main([
        "--config", str(BASE_YAML), "--llm", "--days", "2", "--models", "qwen2.5:7b,gemma3:4b",
        "--runs-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "3qwen2.5:7b+2gemma3:4b" in out and "LLM 호출 20번" in out
    assert "질문지 길이" in out
    (db,) = tmp_path.glob("*_llm-mixed_s1.db")
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT condition, model, status FROM runs").fetchone() == (
        "llm-mixed", "3qwen2.5:7b+2gemma3:4b", "days",
    )
    models = [m for (m,) in conn.execute("SELECT model FROM llm_calls WHERE day = 1 AND phase = 'fish' ORDER BY rowid")]
    assert models == ["qwen2.5:7b", "gemma3:4b", "qwen2.5:7b", "gemma3:4b", "qwen2.5:7b"]
    conn.close()


# ---------------------------------------------------------------- 기억 + 일기 (P5)

DIARY = (
    '{"summary": "' + "연못을 나눠 쓰고 있다. " * 40 + '", '
    '"beliefs": {"estimated_stock": 90, "confidence": 0.7, "sustainable_guess": 10}, '
    '"trust": {"어부2": 0.8, "어부3": 0.6}}'
)


def test_bot_village_writes_no_diary_and_memory_is_still_kept():
    world = make_world()
    result = run(world, all_bots(CooperativeBot), seed=1, days=3)
    assert all(d.diaries == [] for d in result.days)
    a = world.get_agent("a1")
    assert [m.day for m in a.memory] == [1, 2, 3] and a.beliefs is None


def test_diary_updates_beliefs_and_is_logged(tmp_path):
    models = {i: "qwen2.5:7b" for i in IDS}
    conn, _, go = _llm_run(tmp_path, [FISH10], models, days=2, diary=DIARY)
    result = go()
    assert len(result.days[0].diaries) == 5
    beliefs, trust = conn.execute(
        "SELECT beliefs_json, trust_json FROM agent_days WHERE day = 1 AND agent_id = 'a1'"
    ).fetchone()
    assert '"estimated_stock": 90' in beliefs and '"a2": 0.8' in trust
    row = conn.execute("SELECT action_json, clamped FROM actions WHERE phase = 'diary' AND day = 1 LIMIT 1").fetchone()
    assert "연못을 나눠" in row[0] and row[1] == 1  # 요약이 300자를 넘어 잘렸다
    conn.close()


def test_second_morning_prompt_has_yesterday_and_summary(tmp_path):
    models = {i: "qwen2.5:7b" for i in IDS}
    conn, _, go = _llm_run(tmp_path, [FISH10], models, days=2, diary=DIARY)
    go()
    (day1,) = conn.execute("SELECT prompt FROM llm_calls WHERE day = 1 AND phase = 'fish' LIMIT 1").fetchone()
    (day2,) = conn.execute("SELECT prompt FROM llm_calls WHERE day = 2 AND phase = 'fish' LIMIT 1").fetchone()
    assert "최근 기록" not in day1
    for piece in ["1일째: 아침 추정치", "10마리를 원했고 10마리를 잡았다", "지금까지의 요약", "약 90마리"]:
        assert piece in day2, piece
    conn.close()


def test_departed_fisher_writes_no_diary():
    class Silent(BaseAgent):
        def decide_fish(self, obs):
            return {"action": "fish", "amount": 0, "reason": "쉼"}

        def write_diary(self, ctx):
            return {"summary": "s", "beliefs": {"estimated_stock": 1, "confidence": 0, "sustainable_guess": 1}}

    world = make_world(economy={"start_money": 0, "bankrupt_after_days": 1})
    result = run(world, {i: Silent(i) for i in IDS}, seed=1, days=2)
    assert result.end_reason == END_ALL_LEFT
    assert result.days[0].diaries == []  # 1일째 모두 떠나 일기를 쓸 사람이 없다


def test_prompt_length_stops_growing_over_ten_days(tmp_path):
    """P5 완료 기준: 10일 실행에서 질문지 길이(프롬프트 토큰)가 날마다 계속 늘지 않는다.

    가짜 모델이 늘 10마리 + 같은 일기로 답해 연못이 무너지지 않는다. 토큰 대신 보낸 글자 수를 센다.
    """
    models = {i: "qwen2.5:7b" for i in IDS}
    config = llm_config()
    world = World.from_config(config)
    conn = open_db(tmp_path / "len.db")
    logger = RunLogger.start(conn, "r", config_name="t", condition="llm", seed=1, model="qwen2.5:7b")
    client = FakeOllama([FISH10], diary=DIARY)
    client.prompt_chars = True
    caller = llm.LLMCaller(client=client, meter=llm.CostMeter(1.0), record=logger.log_llm_call)
    result = run(world, build_llm_agents(config, caller, models), seed=1, days=10, logger=logger)
    assert result.end_reason == END_DAYS and len(result.days) == 10

    for phase in ("fish", "diary"):
        per_day = [m for (m,) in conn.execute(
            "SELECT MAX(prompt_tokens) FROM llm_calls WHERE phase = ? GROUP BY day ORDER BY day", (phase,)
        )]
        assert len(per_day) == 10
        assert per_day[3] > per_day[0]  # 기억이 붙어 1일째보다는 길다
        settled = per_day[3:]  # 4일째부터: 원본 2일 + 요약 1개가 다 찬 뒤
        assert max(settled) - min(settled) <= 10, (phase, per_day)  # 숫자 자릿수 차이만
    conn.close()
