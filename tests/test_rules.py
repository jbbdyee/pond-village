"""rules.py 테스트. 어획 배분, 판매·생활비, 파산, 재생·붕괴."""

import pytest

from pondvillage.rules import (
    allocate_catch,
    check_bankruptcy,
    check_collapse,
    regenerate,
    remove_catch,
    settle_money,
)
from pondvillage.world import World, config_from_dict

# SPEC 숫자 그대로: 시작 돈 20, 1마리 1, 생활비 5, 빚 3일이면 떠남
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
    "agents": [
        {"id": "a1", "name": "가", "persona": "어부 가"},
        {"id": "a2", "name": "나", "persona": "어부 나"},
        {"id": "a3", "name": "다", "persona": "어부 다"},
    ],
}


def make_world() -> World:
    return World.from_config(config_from_dict(RAW))


def end_of_day(world: World, catches: dict[str, int]) -> list[str]:
    """정산 순서대로: 판매·생활비 -> 파산 판정."""
    settle_money(world, catches)
    return check_bankruptcy(world)

# ---------------------------------------------------------------- 어획 배분


def test_requests_within_stock_are_given_as_asked():
    requests = {"a1": 10, "a2": 10, "a3": 10, "a4": 10, "a5": 10}

    assert allocate_catch(requests, stock=100) == requests


def test_requests_exactly_equal_to_stock_are_given_as_asked():
    requests = {"a1": 20, "a2": 20, "a3": 10}

    assert allocate_catch(requests, stock=50) == requests


def test_everyone_greedy_shares_equally():
    # 재고 50에 다섯 명이 20마리씩(합계 100) 요청 -> 절반씩만 받는다
    requests = {f"a{i}": 20 for i in range(1, 6)}

    assert allocate_catch(requests, stock=50) == {f"a{i}": 10 for i in range(1, 6)}


def test_shortage_is_split_by_request_ratio_rounded_down():
    # 합계 35, 재고 10 -> 20*10/35=5.71, 10*10/35=2.85, 5*10/35=1.42 -> 내림
    result = allocate_catch({"a1": 20, "a2": 10, "a3": 5}, stock=10)

    assert result == {"a1": 5, "a2": 2, "a3": 1}
    assert sum(result.values()) == 8  # 남은 2마리는 아무도 받지 않는다


def test_asking_more_never_gets_less_than_asking_less():
    result = allocate_catch({"a1": 20, "a2": 7, "a3": 1}, stock=9)

    assert result["a1"] >= result["a2"] >= result["a3"]


def test_empty_pond_gives_nothing():
    assert allocate_catch({"a1": 20, "a2": 5}, stock=0) == {"a1": 0, "a2": 0}


def test_zero_requests_get_zero():
    # 연못을 확인만 했거나 쉬는 어부는 0으로 들어온다
    assert allocate_catch({"a1": 0, "a2": 0}, stock=100) == {"a1": 0, "a2": 0}
    assert allocate_catch({"a1": 0, "a2": 20, "a3": 20}, stock=10) == {"a1": 0, "a2": 5, "a3": 5}


def test_no_fishers_returns_empty():
    assert allocate_catch({}, stock=100) == {}


@pytest.mark.parametrize("stock", range(0, 101))
def test_never_hands_out_more_than_stock_or_more_than_asked(stock):
    requests = {"a1": 20, "a2": 13, "a3": 7, "a4": 1, "a5": 0}

    result = allocate_catch(requests, stock)

    assert sum(result.values()) <= stock
    assert all(0 <= result[k] <= requests[k] for k in requests)
    assert result.keys() == requests.keys()


def test_does_not_change_the_input():
    requests = {"a1": 20, "a2": 20}

    allocate_catch(requests, stock=10)

    assert requests == {"a1": 20, "a2": 20}


def test_negative_request_is_rejected():
    with pytest.raises(ValueError):
        allocate_catch({"a1": -1}, stock=10)


def test_negative_stock_is_rejected():
    with pytest.raises(ValueError):
        allocate_catch({"a1": 1}, stock=-1)


# ---------------------------------------------------------------- 판매·생활비


def test_selling_adds_fish_times_price_minus_daily_cost():
    world = make_world()

    settle_money(world, {"a1": 10, "a2": 20, "a3": 3})

    # 20 + 10*1 - 5 = 25, 20 + 20 - 5 = 35, 20 + 3 - 5 = 18
    assert [a.money for a in world.agents] == [25, 35, 18]


def test_fisher_missing_from_catches_still_pays_daily_cost():
    world = make_world()

    settle_money(world, {"a1": 10})

    assert [a.money for a in world.agents] == [25, 15, 15]


def test_money_can_go_below_zero():
    world = make_world()
    world.agents[0].money = 2

    settle_money(world, {"a1": 0})

    assert world.agents[0].money == -3


def test_departed_fisher_pays_nothing():
    world = make_world()
    world.agents[1].alive = False
    world.agents[1].money = -15

    settle_money(world, {})

    assert world.agents[1].money == -15


def test_catch_for_departed_fisher_is_rejected():
    world = make_world()
    world.agents[1].alive = False

    with pytest.raises(ValueError):
        settle_money(world, {"a2": 5})


def test_catch_for_unknown_fisher_is_rejected():
    with pytest.raises(ValueError):
        settle_money(make_world(), {"zz": 5})


def test_negative_catch_is_rejected():
    with pytest.raises(ValueError):
        settle_money(make_world(), {"a1": -1})


def test_ten_fish_a_day_earns_five_a_day():
    world = make_world()

    for _ in range(4):
        end_of_day(world, {"a1": 10, "a2": 10, "a3": 10})

    assert all(a.money == 40 for a in world.agents)


# ---------------------------------------------------------------- 파산 (3일)


def test_fisher_who_catches_nothing_leaves_on_day_seven():
    # 20 -> 15 -> 10 -> 5 -> 0 -> -5(1일째) -> -10(2일째) -> -15(3일째, 떠남)
    world = make_world()
    history = []

    for day in range(1, 8):
        left = end_of_day(world, {"a2": 10, "a3": 10})
        a1 = world.agents[0]
        history.append((day, a1.money, a1.debt_days, a1.alive, left))

    assert history == [
        (1, 15, 0, True, []),
        (2, 10, 0, True, []),
        (3, 5, 0, True, []),
        (4, 0, 0, True, []),
        (5, -5, 1, True, []),
        (6, -10, 2, True, []),
        (7, -15, 3, False, ["a1"]),
    ]
    assert world.agents[1].alive and world.agents[2].alive


def test_zero_money_is_not_debt():
    world = make_world()
    for a in world.agents:
        a.money = 0
        a.debt_days = 2

    assert check_bankruptcy(world) == []
    assert all(a.debt_days == 0 and a.alive for a in world.agents)


def test_two_debt_days_then_recovery_resets_the_count():
    world = make_world()
    a1 = world.agents[0]
    a1.money = 0

    end_of_day(world, {"a1": 0})  # -5, 1일째
    end_of_day(world, {"a1": 0})  # -10, 2일째
    assert a1.debt_days == 2

    end_of_day(world, {"a1": 20})  # -10 + 20 - 5 = 5 -> 회복
    assert (a1.money, a1.debt_days, a1.alive) == (5, 0, True)

    # 다시 빚을 져도 처음부터 센다: 이틀 더 버틴다
    a1.money = 0
    end_of_day(world, {"a1": 0})
    end_of_day(world, {"a1": 0})
    assert a1.alive and a1.debt_days == 2


def test_several_fishers_can_leave_on_the_same_day():
    world = make_world()
    for a in world.agents:
        a.money = -1
        a.debt_days = 2

    world.agents[1].money = 3  # a2만 빚이 없다

    assert check_bankruptcy(world) == ["a1", "a3"]
    assert [a.alive for a in world.agents] == [False, True, False]


def test_departed_fisher_is_not_counted_again():
    world = make_world()
    a1 = world.agents[0]
    a1.money, a1.debt_days, a1.alive = -15, 3, False

    assert check_bankruptcy(world) == []
    assert a1.debt_days == 3


# ---------------------------------------------------------------- 연못: 빼기·붕괴·재생


def run_pond_day(world: World, requests: dict[str, int]) -> bool:
    """정산 순서대로 하루: 배분 -> 연못에서 빼기 -> 판매·생활비 -> 파산 -> 붕괴 판정 -> 재생."""
    catches = allocate_catch(requests, world.pond.stock)
    remove_catch(world, catches)
    end_of_day(world, catches)
    collapsed = check_collapse(world)
    regenerate(world)
    return collapsed


def pond_end_of_day(world: World, stock_left: int) -> int:
    """남은 물고기를 정해 두고 붕괴 판정 -> 재생만 돌려 다음 날 아침 재고를 본다."""
    world.pond.stock = stock_left
    check_collapse(world)
    regenerate(world)
    return world.pond.stock


def test_remove_catch_subtracts_total_from_stock():
    world = make_world()
    remove_catch(world, {"a1": 10, "a2": 20, "a3": 0})
    assert world.pond.stock == 70


def test_remove_catch_can_empty_the_pond():
    world = make_world()
    remove_catch(world, {"a1": 50, "a2": 50})
    assert world.pond.stock == 0


def test_remove_catch_more_than_stock_is_rejected():
    world = make_world()
    with pytest.raises(ValueError):
        remove_catch(world, {"a1": 60, "a2": 41})
    assert world.pond.stock == 100


def test_remove_negative_catch_is_rejected():
    world = make_world()
    with pytest.raises(ValueError):
        remove_catch(world, {"a1": -1})


@pytest.mark.parametrize(
    ("left", "next_morning"),
    [(5, 10), (30, 60), (50, 100), (60, 100), (100, 100)],
)
def test_regeneration_doubles_up_to_capacity(left, next_morning):
    world = make_world()
    assert pond_end_of_day(world, left) == next_morning
    assert world.pond.collapsed is False


@pytest.mark.parametrize("left", [0, 1, 4])
def test_fewer_than_five_left_collapses_without_regeneration(left):
    world = make_world()
    assert pond_end_of_day(world, left) == left
    assert world.pond.collapsed is True


def test_exactly_five_left_does_not_collapse():
    world = make_world()
    assert check_collapse(world) is False
    world.pond.stock = 5
    assert check_collapse(world) is False


def test_collapse_is_permanent():
    world = make_world()
    pond_end_of_day(world, 3)
    world.pond.stock = 50  # 누가 억지로 채워 넣어도
    assert check_collapse(world) is True
    regenerate(world)
    assert world.pond.stock == 50


def test_greedy_village_collapses_on_day_one():
    """SPEC 숫자 그대로 5명이 매일 20마리씩: 첫날 100마리를 다 잡아 붕괴."""
    raw = {**RAW, "agents": [{"id": f"g{i}", "name": f"{i}", "persona": "탐욕"} for i in range(5)]}
    world = World.from_config(config_from_dict(raw))
    assert run_pond_day(world, {a.id: 20 for a in world.agents}) is True
    assert world.pond.stock == 0


def test_three_greedy_fishers_collapse_on_day_three():
    """3명이 매일 20마리: 100->40->80, 80->20->40, 40에서 13씩 39마리 -> 1마리 남아 붕괴."""
    world = make_world()
    requests = {"a1": 20, "a2": 20, "a3": 20}
    assert run_pond_day(world, requests) is False
    assert world.pond.stock == 80
    assert run_pond_day(world, requests) is False
    assert world.pond.stock == 40
    assert run_pond_day(world, requests) is True
    assert world.pond.stock == 1


def test_sustainable_fishing_keeps_pond_full_for_30_days():
    """잡고 남은 게 절반 이상이면 재생으로 다시 100. 30일 내내 유지되고 아무도 떠나지 않는다."""
    world = make_world()
    for _ in range(30):
        assert run_pond_day(world, {"a1": 10, "a2": 10, "a3": 10}) is False
        assert world.pond.stock == 100
    assert all(a.alive for a in world.agents)


# ---------------------------------------------------------------- 설정 숫자 따르기·정산 전체 흐름


def make_custom_world(pond: dict | None = None, economy: dict | None = None) -> World:
    """RAW에서 일부 숫자만 바꾼 마을. 규칙이 숫자를 코드에 박아 두지 않았는지 볼 때 쓴다."""
    raw = {
        **RAW,
        "pond": {**RAW["pond"], **(pond or {})},
        "economy": {**RAW["economy"], **(economy or {})},
    }
    return World.from_config(config_from_dict(raw))


@pytest.mark.parametrize(
    ("left", "next_morning", "collapsed"),
    [(15, 45, False), (25, 60, False), (10, 30, False), (9, 9, True)],
)
def test_pond_rules_follow_config_numbers(left, next_morning, collapsed):
    """연못 크기 60, 재생 3배, 10마리 미만이면 붕괴로 바꾸면 규칙도 그 숫자대로 움직인다."""
    world = make_custom_world(
        pond={"capacity": 60, "initial_stock": 60, "regen_multiplier": 3, "collapse_below": 10}
    )
    assert pond_end_of_day(world, left) == next_morning
    assert world.pond.collapsed is collapsed


def test_economy_rules_follow_config_numbers():
    """시작 돈 4, 1마리 2, 생활비 3, 빚 2일이면 떠남으로 바꾸면 그 숫자대로 정산한다."""
    world = make_custom_world(
        economy={"start_money": 4, "fish_price": 2, "daily_cost": 3, "bankrupt_after_days": 2}
    )
    a1, a2 = world.agents[0], world.agents[1]

    # a1은 매일 0마리: 4 -> 1 -> -2(1일째) -> -5(2일째, 떠남)
    assert end_of_day(world, {"a2": 5, "a3": 5}) == []
    assert (a1.money, a2.money) == (1, 11)  # a2: 4 + 5*2 - 3
    assert end_of_day(world, {"a2": 5, "a3": 5}) == []
    assert (a1.money, a1.debt_days) == (-2, 1)
    assert end_of_day(world, {"a2": 5, "a3": 5}) == ["a1"]
    assert (a1.money, a1.debt_days, a1.alive) == (-5, 2, False)


def test_collapse_and_bankruptcy_on_the_same_day():
    """정산 순서(배분 -> 빼기 -> 판매·생활비 -> 파산 -> 붕괴 -> 재생)대로 둘 다 기록된다."""
    world = make_world()
    world.pond.stock = 30
    a1, a2, a3 = world.agents
    a1.money, a1.debt_days = -10, 2  # 오늘 또 빚지면 3일째

    catches = allocate_catch({"a1": 0, "a2": 20, "a3": 20}, world.pond.stock)
    remove_catch(world, catches)
    left = end_of_day(world, catches)
    collapsed = check_collapse(world)
    regenerate(world)

    assert catches == {"a1": 0, "a2": 15, "a3": 15}
    assert left == ["a1"] and a1.alive is False and a1.money == -15
    assert a2.money == a3.money == 30  # 20 + 15 - 5: 붕괴한 날에도 잡은 건 판다
    assert collapsed is True
    assert world.pond.stock == 0  # 붕괴했으니 재생하지 않는다


def run_30_days(world: World, wants: dict[str, int]) -> list[tuple]:
    """살아 있는 어부만 요청을 내며 30일을 돈다. 날마다 (재고, 돈, 생존) 기록을 돌려준다."""
    history = []
    for _ in range(30):
        requests = {a.id: wants[a.id] for a in world.alive_agents()}
        run_pond_day(world, requests)
        history.append(
            (
                world.pond.stock,
                tuple(a.money for a in world.agents),
                tuple(a.alive for a in world.agents),
            )
        )
    return history


def test_same_input_gives_same_result():
    """규칙 엔진에는 무작위가 없다: 같은 30일을 두 번 돌리면 날마다 상태가 똑같다."""
    wants = {"a1": 20, "a2": 15, "a3": 0}  # a3는 아무것도 안 잡아 7일째 떠난다

    first = run_30_days(make_world(), wants)
    second = run_30_days(make_world(), wants)

    assert first == second
    assert first[5][2] == (True, True, True)
    assert first[6][2] == (True, True, False)


def test_empty_village_does_not_break_settlement():
    """모두 떠난 마을에서도 정산 함수들은 멈추지 않는다. 아무도 안 잡았으니 연못은 그대로."""
    world = make_world()
    for a in world.agents:
        a.alive = False

    assert allocate_catch({}, world.pond.stock) == {}
    assert check_bankruptcy(world) == []
    assert run_pond_day(world, {}) is False
    assert world.pond.stock == 100


def test_free_rider_is_not_noticed_by_the_pond():
    """한 명이 20, 두 명이 15씩(합계 50): 연못은 30일 내내 100이고, 많이 잡은 사람만 돈이 더 쌓인다."""
    world = make_world()
    for _ in range(30):
        assert run_pond_day(world, {"a1": 20, "a2": 15, "a3": 15}) is False
        assert world.pond.stock == 100

    assert [a.money for a in world.agents] == [20 + 15 * 30, 20 + 10 * 30, 20 + 10 * 30]
