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
    run_day,
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
    assert main(["--config", str(BASE_YAML), "--days", "2", "--bot", "coop"]) == 0
    out = capsys.readouterr().out
    for name in [a.name for a in load_config(BASE_YAML).agents]:
        assert name in out
    assert "정한 날짜까지 버팀 (2일 진행)" in out


def test_cli_greedy_reports_collapse(capsys):
    main(["--config", str(BASE_YAML), "--bot", "greedy"])
    out = capsys.readouterr().out
    assert "붕괴" in out and "(1일 진행)" in out
