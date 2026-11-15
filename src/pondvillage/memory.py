"""기억 저장·요약 (SPEC 5장).

어부의 기억은 두 종류다 (`world.MemoryItem.kind`).
- 사실(event): 그날 본 공지, 내 행동과 결과. 코드가 정산 뒤에 적는다 (LLM이 지어내지 않는다).
- 일기(diary): 저녁 일기의 summary. LLM이 "지금까지의 요약"을 매번 새로 고쳐 쓴다.

질문지에 넣는 것은 늘 같은 크기다: 최근 2일 사실 원본 + 가장 최근 일기 요약 1개(최대 300자)
+ 지난 일기의 내 생각(beliefs) 한 줄. 날이 지나도 이 묶음은 커지지 않는다.
(요약을 날마다 쌓으면 30일에 28줄이 되어 결국 계속 늘어서, 하나를 고쳐 쓰게 했다. 2026-11-15 사용자 승인)

광장(P6): 모두에게 한 말은 남은 어부 전원이, 귓속말은 보낸 사람과 받는 사람만 듣는다(SPEC 3장).
들은 말은 그날 사실 기억에 한 줄로 들어간다 (말한 사람 이름 + 누구에게 + 말).

신뢰(trust)는 일기에서 받는다. 질문지에는 대화를 켠 마을에서만 넣는다(SPEC 5장, P6부터).
대화를 끈 마을은 이웃에 대해 알 길이 없어 근거 없는 숫자라서, P5와 같은 질문지를 그대로 쓴다.

판정(몇 마리를 받는지 등)은 여기서 하지 않는다. 일기 답의 모양 검사·범위 자르기만 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pondvillage.actions import NONE, PARSE_ERROR_KEY, TO_ALL, TO_SELF
from pondvillage.world import Agent, Beliefs, MemoryItem

KIND_EVENT = "event"
KIND_DIARY = "diary"

RECENT_DAYS = 2  # 원본을 그대로 넣는 최근 날 수 (SPEC 5장)
SUMMARY_MAX_CHARS = 300  # 일기 요약 최대 글자 수. 넘으면 코드가 자른다 (SPEC 5장)


# ---------------------------------------------------------------- 기억 적기


def describe_day(
    day: int,
    estimate: int,
    yesterday_total: int | None,
    *,
    kind: str,
    amount: int,
    caught: int,
    exact_stock: int | None,
    parse_error: bool,
    earned: int,
    cost: int,
    money: int,
    left_names: list[str],
) -> str:
    """어부 한 명이 그날 겪은 사실을 한 줄로. 남이 몇 마리 잡았는지는 넣지 않는다 (모르는 정보)."""
    parts = [f"{day}일째: 아침 추정치 약 {estimate}마리"]
    if yesterday_total is not None:
        parts[0] += f", 어제 마을 전체 어획 {yesterday_total}마리"
    parts[0] += "."
    if exact_stock is not None:
        parts.append(f"연못을 확인해 정확한 재고 {exact_stock}마리를 봤다.")
    if parse_error:
        parts.append("답을 제대로 하지 못해 아무것도 하지 않았다.")
    elif kind == NONE or amount == 0:
        parts.append("물고기를 잡지 않았다.")
    else:
        parts.append(f"{amount}마리를 원했고 {caught}마리를 잡았다.")
    parts.append(f"판 돈 {earned}원, 생활비 {cost}원, 남은 돈 {money}원.")
    if left_names:
        parts.append(f"오늘 마을을 떠난 사람: {', '.join(left_names)}.")
    return " ".join(parts)


# ---------------------------------------------------------------- 광장에서 들은 말 (P6)


@dataclass(frozen=True)
class PlazaMessage:
    """광장에서 나온 말 하나 (검사를 마친 뒤). to는 TO_ALL 또는 받는 어부 id."""

    from_id: str
    to: str
    text: str

    def heard_by(self, agent_id: str) -> bool:
        """이 어부가 이 말을 들었는지. 귓속말은 보낸 사람과 받는 사람만 안다."""
        if self.to == TO_SELF:  # 혼잣말: 말한 사람만
            return agent_id == self.from_id
        return self.to == TO_ALL or agent_id in (self.to, self.from_id)


def heard_lines(messages: list[PlazaMessage], agent_id: str, names: dict[str, str]) -> list[str]:
    """이 어부가 들은 말만, 말한 순서대로. 예: 순자(모두에게): "..." / 나(영철에게 귓속말): "..." """
    lines = []
    for m in messages:
        if not m.heard_by(agent_id):
            continue
        who = "나" if m.from_id == agent_id else names[m.from_id]
        if m.to == TO_ALL:
            where = "모두에게"
        elif m.to == TO_SELF:
            where = "혼잣말, 아무도 듣지 않음"
        elif m.to == agent_id:
            where = "나에게 귓속말"
        else:
            where = f"{names[m.to]}에게 귓속말"
        lines.append(f'{who}({where}): "{m.text}"')
    return lines


def heard_text(messages: list[PlazaMessage], agent_id: str, names: dict[str, str]) -> str:
    """질문지에 붙일 "오늘 광장에서 들은 말" 글. 들은 말이 없으면 빈 글."""
    return "\n".join(heard_lines(messages, agent_id, names))


def describe_plaza(day: int, messages: list[PlazaMessage], agent_id: str, names: dict[str, str]) -> str | None:
    """그날 들은 말을 기억 한 줄로. 들은 말이 없으면 None (기억에 적지 않는다)."""
    lines = heard_lines(messages, agent_id, names)
    return f"{day}일째 광장: " + " / ".join(lines) if lines else None


def trust_line(trust: dict[str, float], names: dict[str, str]) -> str:
    """신뢰 점수를 이름으로 한 줄. names에 있는 사람(남은 이웃)만. 예: 순자 0.8, 영철 0.5"""
    return ", ".join(f"{names[i]} {v:.1f}" for i, v in trust.items() if i in names)


def remember(agent: Agent, day: int, text: str) -> None:
    agent.memory.append(MemoryItem(day, text, KIND_EVENT))


def remember_summary(agent: Agent, day: int, summary: str) -> None:
    agent.memory.append(MemoryItem(day, summary, KIND_DIARY))


# ---------------------------------------------------------------- 질문지용 기억 글


def latest_summary(agent: Agent, before_or_on: int) -> MemoryItem | None:
    diaries = [m for m in agent.memory if m.kind == KIND_DIARY and m.day <= before_or_on]
    return diaries[-1] if diaries else None


def beliefs_line(beliefs: Beliefs) -> str:
    return (
        f"연못에 약 {beliefs.estimated_stock}마리가 있을 것 같다(확신 {beliefs.confidence:.1f}). "
        f"한 사람이 하루 {beliefs.sustainable_guess}마리까지는 괜찮을 것 같다."
    )


def memory_text(
    agent: Agent,
    upto_day: int,
    recent_days: int = RECENT_DAYS,
    trust_names: dict[str, str] | None = None,
) -> str:
    """질문지에 붙일 기억 글. upto_day까지 있었던 일만 본다.

    어획(아침) 때는 upto_day = 어제, 일기(저녁) 때는 upto_day = 오늘.
    원본은 (upto_day - recent_days, upto_day] 범위만, 그 전은 일기 요약 하나로만 들어간다.
    trust_names(남은 이웃 id → 이름)를 주면 지난 일기의 신뢰 점수도 한 줄 붙인다 (대화를 켠 마을만).
    기억이 하나도 없으면(1일째 아침) 빈 글.
    """
    blocks: list[str] = []
    summary = latest_summary(agent, upto_day)
    if summary is not None:
        blocks.append(f"[지금까지의 요약 ({summary.day}일째 일기)]\n{summary.text}")
    recent = [m for m in agent.memory if m.kind == KIND_EVENT and upto_day - recent_days < m.day <= upto_day]
    if recent:
        blocks.append("[최근 기록]\n" + "\n".join(m.text for m in recent))
    if agent.beliefs is not None:
        blocks.append("[지난 일기에 적은 내 생각]\n" + beliefs_line(agent.beliefs))
    if trust_names is not None:
        line = trust_line(agent.trust, trust_names)
        if line:
            blocks.append("[지난 일기에 적은 이웃 신뢰 (0~1)]\n" + line)
    return "\n".join(blocks) + "\n" if blocks else ""


# ---------------------------------------------------------------- 일기 답 검사


@dataclass(frozen=True)
class DiaryEntry:
    """검사를 마친 일기 하나. parse_error면 summary·beliefs·trust는 None (이전 것을 그대로 둔다)."""

    agent_id: str
    summary: str | None
    beliefs: Beliefs | None
    trust: dict[str, float] | None
    clamped: bool = False  # 범위를 벗어난 값을 잘랐거나 요약이 길어 잘렸는지
    parse_error: str | None = None


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _clip(value: float, low: float, high: float) -> tuple[float, bool]:
    clipped = min(max(value, low), high)
    return clipped, clipped != value


def validate_diary(
    raw: object,
    agent_id: str,
    name_to_id: dict[str, str],
    *,
    capacity: int,
    max_catch: int,
) -> DiaryEntry:
    """일기 답 검사. summary(글)와 beliefs 세 숫자가 있어야 한다. 아니면 parse_error.

    - estimated_stock은 0~연못 최대, sustainable_guess는 0~하루 상한, confidence·trust는 0~1로 자른다.
    - trust는 이름으로 받아 id로 바꾼다. 모르는 이름·자기 자신·숫자 아닌 값은 버린다 (실패로 보지 않음).
    - summary는 SUMMARY_MAX_CHARS에서 자른다.
    """
    def fail(why: str) -> DiaryEntry:
        return DiaryEntry(agent_id, None, None, None, parse_error=why)

    if not isinstance(raw, dict):
        return fail(f"dict가 아님: {type(raw).__name__}")
    if raw.get(PARSE_ERROR_KEY):
        return fail(str(raw[PARSE_ERROR_KEY]))

    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return fail(f"summary가 글이 아님: {summary!r}")
    summary = summary.strip()
    clamped = len(summary) > SUMMARY_MAX_CHARS
    summary = summary[:SUMMARY_MAX_CHARS]

    b = raw.get("beliefs")
    if not isinstance(b, dict):
        return fail(f"beliefs가 dict가 아님: {b!r}")
    stock = _as_number(b.get("estimated_stock"))
    conf = _as_number(b.get("confidence"))
    guess = _as_number(b.get("sustainable_guess"))
    if stock is None or conf is None or guess is None:
        return fail(f"beliefs 숫자가 빠졌거나 숫자가 아님: {b!r}")
    stock, c1 = _clip(round(stock), 0, capacity)
    conf, c2 = _clip(conf, 0.0, 1.0)
    guess, c3 = _clip(round(guess), 0, max_catch)
    clamped = clamped or c1 or c2 or c3
    beliefs = Beliefs(estimated_stock=int(stock), confidence=conf, sustainable_guess=int(guess))

    trust: dict[str, float] = {}
    raw_trust = raw.get("trust")
    if isinstance(raw_trust, dict):
        for name, value in raw_trust.items():
            other = name_to_id.get(str(name).strip())
            number = _as_number(value)
            if other is None or other == agent_id or number is None:
                continue
            trust[other], c = _clip(number, 0.0, 1.0)
            clamped = clamped or c
    return DiaryEntry(agent_id, summary, beliefs, trust, clamped=clamped)


def apply_diary(agent: Agent, day: int, entry: DiaryEntry) -> None:
    """검사한 일기를 어부에게 반영한다. 실패한 일기는 아무것도 바꾸지 않는다 (지난 생각 유지)."""
    if entry.parse_error is not None:
        return
    assert entry.summary is not None
    remember_summary(agent, day, entry.summary)
    agent.beliefs = entry.beliefs
    if entry.trust:
        agent.trust = {**agent.trust, **entry.trust}
