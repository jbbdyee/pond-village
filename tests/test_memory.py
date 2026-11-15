"""memory.py 테스트: 기억 적기, 질문지용 기억 글(최근 2일 + 요약), 일기 답 검사."""

from pondvillage.actions import FISH, NONE, PARSE_ERROR_KEY
from pondvillage.memory import (
    KIND_DIARY, KIND_EVENT, SUMMARY_MAX_CHARS, apply_diary, describe_day, memory_text, remember,
    remember_summary, validate_diary,
)
from pondvillage.world import Agent, Beliefs

NAMES = {"만석": "agent_01", "순자": "agent_02", "영철": "agent_03"}


def agent():
    return Agent(id="agent_01", name="만석", persona="", money=20)


def day_text(day, **kw):
    base = dict(kind=FISH, amount=10, caught=10, exact_stock=None, parse_error=False,
                earned=10, cost=5, money=25, left_names=[])
    base.update(kw)
    return describe_day(day, 95, 50 if day > 1 else None, **base)


def check(raw, **kw):
    return validate_diary(raw, "agent_01", NAMES, capacity=100, max_catch=20, **kw)


GOOD = {
    "summary": "첫날 10마리를 잡았다.",
    "beliefs": {"estimated_stock": 80, "confidence": 0.6, "sustainable_guess": 10},
    "trust": {"순자": 0.7, "영철": 0.5},
}


# ---------------------------------------------------------------- 기억 적기


def test_describe_day_has_only_my_facts():
    text = day_text(2)
    for piece in ["2일째", "추정치 약 95마리", "어제 마을 전체 어획 50마리", "10마리를 원했고 10마리를 잡았다",
                  "판 돈 10원", "생활비 5원", "남은 돈 25원"]:
        assert piece in text, piece


def test_describe_day_first_day_has_no_yesterday():
    assert "어제" not in day_text(1)


def test_describe_day_inspect_parse_error_none_and_departures():
    assert "정확한 재고 61마리" in day_text(3, exact_stock=61)
    assert "아무것도 하지 않았다" in day_text(3, kind=NONE, amount=0, caught=0, parse_error=True)
    assert "잡지 않았다" in day_text(3, kind=NONE, amount=0, caught=0)
    assert "떠난 사람: 영철" in day_text(3, left_names=["영철"])


def test_remember_adds_event_items():
    a = agent()
    remember(a, 1, "하루")
    remember_summary(a, 1, "요약")
    assert [(m.day, m.kind) for m in a.memory] == [(1, KIND_EVENT), (1, KIND_DIARY)]


# ---------------------------------------------------------------- 질문지용 기억 글


def test_first_morning_has_no_memory():
    assert memory_text(agent(), 0) == ""


def test_only_last_two_days_raw_and_latest_summary():
    a = agent()
    for d in range(1, 6):
        remember(a, d, f"기록{d}")
        remember_summary(a, d, f"요약{d}")
    text = memory_text(a, 5)
    assert "기록4" in text and "기록5" in text
    assert "기록3" not in text and "기록1" not in text
    assert "요약5" in text and "요약4" not in text


def test_morning_view_does_not_see_today():
    a = agent()
    remember(a, 4, "기록4")
    remember(a, 5, "기록5")
    remember_summary(a, 5, "요약5")
    text = memory_text(a, 4)  # 5일째 아침이 아니라 4일째까지
    assert "기록4" in text and "기록5" not in text and "요약5" not in text


def test_beliefs_line_is_included():
    a = agent()
    a.beliefs = Beliefs(estimated_stock=70, confidence=0.4, sustainable_guess=8)
    text = memory_text(a, 1)
    assert "약 70마리" in text and "하루 8마리" in text and "0.4" in text


def test_memory_text_stays_same_size_as_days_pass():
    """완료 기준의 핵심: 같은 길이 기록이면 3일째 이후 기억 글 길이가 늘지 않는다."""
    a = agent()
    lengths = []
    for d in range(1, 31):
        remember(a, d, day_text(d))
        remember_summary(a, d, "가" * SUMMARY_MAX_CHARS)
        lengths.append(len(memory_text(a, d)))
    assert max(lengths[3:]) - min(lengths[3:]) <= 4  # 날짜 숫자 자릿수 차이만


# ---------------------------------------------------------------- 일기 답 검사


def test_good_diary_passes_and_maps_names_to_ids():
    e = check(GOOD)
    assert e.parse_error is None and not e.clamped
    assert e.summary == "첫날 10마리를 잡았다."
    assert e.beliefs == Beliefs(estimated_stock=80, confidence=0.6, sustainable_guess=10)
    assert e.trust == {"agent_02": 0.7, "agent_03": 0.5}


def test_out_of_range_values_are_clamped():
    raw = {"summary": "s", "beliefs": {"estimated_stock": 150, "confidence": 1.5, "sustainable_guess": 30},
           "trust": {"순자": -0.2}}
    e = check(raw)
    assert e.clamped
    assert e.beliefs == Beliefs(estimated_stock=100, confidence=1.0, sustainable_guess=20)
    assert e.trust == {"agent_02": 0.0}


def test_long_summary_is_cut():
    e = check({**GOOD, "summary": "가" * (SUMMARY_MAX_CHARS + 50)})
    assert len(e.summary) == SUMMARY_MAX_CHARS and e.clamped


def test_unknown_names_self_and_bad_trust_values_are_dropped():
    e = check({**GOOD, "trust": {"철수": 0.9, "만석": 1.0, "순자": "높음", "영철": 0.3}})
    assert e.parse_error is None
    assert e.trust == {"agent_03": 0.3}


def test_missing_trust_is_fine():
    e = check({k: v for k, v in GOOD.items() if k != "trust"})
    assert e.parse_error is None and e.trust == {}


def test_bad_shapes_are_parse_errors():
    assert check("일기").parse_error
    assert check({PARSE_ERROR_KEY: "두 번 모두 해석 실패"}).parse_error == "두 번 모두 해석 실패"
    assert check({"action": "fish", "amount": 10}).parse_error  # 어획 답을 잘못 준 경우
    assert check({**GOOD, "summary": "  "}).parse_error
    assert check({**GOOD, "beliefs": "모름"}).parse_error
    assert check({**GOOD, "beliefs": {"estimated_stock": "많이", "confidence": 0.5, "sustainable_guess": 5}}).parse_error
    assert check({**GOOD, "beliefs": {"estimated_stock": True, "confidence": 0.5, "sustainable_guess": 5}}).parse_error


def test_apply_diary_updates_agent():
    a = agent()
    a.trust = {"agent_03": 0.1}
    apply_diary(a, 2, check(GOOD))
    assert a.memory[-1].kind == KIND_DIARY and a.memory[-1].day == 2
    assert a.beliefs.estimated_stock == 80
    assert a.trust == {"agent_02": 0.7, "agent_03": 0.5}


def test_failed_diary_keeps_previous_thoughts():
    a = agent()
    apply_diary(a, 1, check(GOOD))
    apply_diary(a, 2, check("엉망"))
    assert len(a.memory) == 1 and a.beliefs.estimated_stock == 80
