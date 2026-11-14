"""actions.py 테스트: 어획 답 검사 (상한 자르기, 확인한 날 절반, 해석 실패)."""

import pytest

from pondvillage.actions import FISH, INSPECT, NONE, catch_cap, validate_fish_action
from pondvillage.world import EconomyConfig


def economy(max_catch: int = 20, factor: float = 0.5) -> EconomyConfig:
    return EconomyConfig(
        start_money=20,
        fish_price=1,
        daily_cost=5,
        max_catch_per_day=max_catch,
        inspect_catch_factor=factor,
        bankrupt_after_days=3,
    )


# ---------------------------------------------------------------- 상한


def test_cap_is_20_normally_and_10_after_inspect():
    assert catch_cap(economy(), inspected=False) == 20
    assert catch_cap(economy(), inspected=True) == 10


def test_cap_follows_config_and_rounds_down():
    # 15 × 0.25 = 3.75 → 3 (상한을 넘겨 주지 않는다)
    assert catch_cap(economy(max_catch=15, factor=0.25), inspected=True) == 3
    assert catch_cap(economy(max_catch=15, factor=0.25), inspected=False) == 15


# ---------------------------------------------------------------- 정상 답


def test_fish_within_cap_passes_unchanged():
    d = validate_fish_action({"action": "fish", "amount": 12, "reason": "적당히"}, cap=20)
    assert (d.kind, d.amount, d.reason, d.clamped, d.parse_error) == (FISH, 12, "적당히", False, None)


@pytest.mark.parametrize("amount", [0, 20])
def test_fish_at_edges_is_not_clamped(amount):
    d = validate_fish_action({"action": "fish", "amount": amount}, cap=20)
    assert d.amount == amount and not d.clamped


def test_whole_float_is_accepted():
    d = validate_fish_action({"action": "fish", "amount": 10.0}, cap=20)
    assert d.kind == FISH and d.amount == 10 and d.parse_error is None


def test_missing_reason_becomes_empty_string():
    assert validate_fish_action({"action": "fish", "amount": 5}, cap=20).reason == ""


# ---------------------------------------------------------------- 자르기 (clamped)


@pytest.mark.parametrize("asked, expected", [(30, 20), (21, 20), (-3, 0)])
def test_out_of_range_is_clamped(asked, expected):
    d = validate_fish_action({"action": "fish", "amount": asked}, cap=20)
    assert d.kind == FISH and d.amount == expected and d.clamped
    assert d.parse_error is None


def test_inspected_day_clamps_to_half():
    d = validate_fish_action({"action": "fish", "amount": 20}, cap=10, allow_inspect=False)
    assert d.amount == 10 and d.clamped


# ---------------------------------------------------------------- 연못 확인


def test_inspect_is_accepted_on_first_answer():
    d = validate_fish_action({"action": "inspect_pond", "reason": "직접 볼래"}, cap=20)
    assert (d.kind, d.amount, d.reason, d.parse_error) == (INSPECT, 0, "직접 볼래", None)


def test_inspect_again_means_do_nothing_without_error():
    d = validate_fish_action({"action": "inspect_pond"}, cap=10, allow_inspect=False)
    assert (d.kind, d.amount, d.parse_error) == (NONE, 0, None)


# ---------------------------------------------------------------- 해석 실패 (parse_error)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "fish 10",
        ["fish", 10],
        {},
        {"action": "say", "message": "안녕"},
        {"action": "fish"},
        {"action": "fish", "amount": "10"},
        {"action": "fish", "amount": 10.5},
        {"action": "fish", "amount": True},
        {"action": "fish", "amount": None},
    ],
)
def test_bad_answer_becomes_do_nothing_with_parse_error(raw):
    d = validate_fish_action(raw, cap=20)
    assert d.kind == NONE and d.amount == 0
    assert d.parse_error  # 무엇이 틀렸는지 적혀 있다
    assert not d.clamped


def test_non_string_reason_is_kept_as_text():
    d = validate_fish_action({"action": "fish", "amount": 3, "reason": 42}, cap=20)
    assert d.reason == "42"
