"""agents/scripted.py 테스트. 탐욕·협력·랜덤 봇."""

import random

import pytest

from pondvillage.agents.base import BaseAgent, Observation
from pondvillage.agents.scripted import CooperativeBot, GreedyBot, RandomBot


def obs(day=1, money=20, estimate=100, yesterday=None):
    return Observation(
        day=day,
        agent_id="agent_01",
        money=money,
        stock_estimate=estimate,
        yesterday_total_catch=yesterday,
    )


def assert_spec_fish(action):
    """SPEC 4장: {"action": "fish", "amount": 0~20, "reason": "..."}"""
    assert set(action) == {"action", "amount", "reason"}
    assert action["action"] == "fish"
    assert isinstance(action["amount"], int)
    assert isinstance(action["reason"], str) and action["reason"]


# ---------------------------------------------------------------- 공통 약속


@pytest.mark.parametrize(
    "bot",
    [GreedyBot("a"), CooperativeBot("a"), RandomBot("a", random.Random(1))],
)
def test_bots_follow_interface_and_spec_shape(bot):
    assert isinstance(bot, BaseAgent)
    assert bot.agent_id == "a"
    assert_spec_fish(bot.decide_fish(obs()))


def test_base_agent_cannot_be_used_directly():
    with pytest.raises(TypeError):
        BaseAgent("a")


def test_observation_has_no_exact_stock():
    # 정보 비대칭: 어부는 정확한 재고를 모른다 (SPEC 1장)
    assert not hasattr(obs(), "stock")


# ---------------------------------------------------------------- 탐욕·협력


def test_greedy_always_20_whatever_it_sees():
    bot = GreedyBot("a")
    for o in [obs(), obs(day=5, money=-3, estimate=8, yesterday=100)]:
        assert bot.decide_fish(o)["amount"] == 20


def test_cooperative_always_10_whatever_it_sees():
    bot = CooperativeBot("a")
    for o in [obs(), obs(day=5, money=-3, estimate=8, yesterday=100)]:
        assert bot.decide_fish(o)["amount"] == 10


def test_fixed_bots_amount_can_be_changed():
    assert GreedyBot("a", amount=15).decide_fish(obs())["amount"] == 15
    assert CooperativeBot("a", amount=7).decide_fish(obs())["amount"] == 7


@pytest.mark.parametrize("cls", [GreedyBot, CooperativeBot])
def test_fixed_bots_reject_negative_amount(cls):
    with pytest.raises(ValueError):
        cls("a", amount=-1)


def test_five_cooperative_bots_take_exactly_sustainable_50():
    # SPEC: 지속 가능한 하루 총 어획량 50 = 5명 × 10
    bots = [CooperativeBot(f"agent_0{i}") for i in range(1, 6)]
    assert sum(b.decide_fish(obs())["amount"] for b in bots) == 50


# ---------------------------------------------------------------- 랜덤


def test_random_stays_in_range():
    bot = RandomBot("a", random.Random(7))
    amounts = [bot.decide_fish(obs(day=d))["amount"] for d in range(1, 501)]
    assert min(amounts) >= 0 and max(amounts) <= 20
    assert min(amounts) == 0 and max(amounts) == 20  # 양 끝도 나온다


def test_random_same_seed_same_sequence():
    a = RandomBot("a", random.Random(42))
    b = RandomBot("a", random.Random(42))
    seq_a = [a.decide_fish(obs(day=d))["amount"] for d in range(1, 31)]
    seq_b = [b.decide_fish(obs(day=d))["amount"] for d in range(1, 31)]
    assert seq_a == seq_b


def test_random_different_seed_different_sequence():
    a = RandomBot("a", random.Random(1))
    b = RandomBot("a", random.Random(2))
    seq_a = [a.decide_fish(obs(day=d))["amount"] for d in range(1, 31)]
    seq_b = [b.decide_fish(obs(day=d))["amount"] for d in range(1, 31)]
    assert seq_a != seq_b


def test_random_does_not_touch_global_random():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    RandomBot("a", random.Random(5)).decide_fish(obs())
    assert random.random() == expected


def test_random_custom_range():
    bot = RandomBot("a", random.Random(3), low=5, high=5)
    assert bot.decide_fish(obs())["amount"] == 5


@pytest.mark.parametrize("low,high", [(-1, 20), (10, 9)])
def test_random_rejects_bad_range(low, high):
    with pytest.raises(ValueError):
        RandomBot("a", random.Random(1), low=low, high=high)
