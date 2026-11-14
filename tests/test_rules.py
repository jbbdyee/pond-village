"""rules.py 테스트. 지금은 어획 배분만 있다 (판매·파산·재생은 다음 항목에서 추가)."""

import pytest

from pondvillage.rules import allocate_catch

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
