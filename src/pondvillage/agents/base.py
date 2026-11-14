"""에이전트 인터페이스: 모든 어부(봇이든 LLM이든)가 지키는 공통 약속.

약속은 하나다: 아침에 본 것(`Observation`)을 받으면 어획 행동을 dict로 돌려준다.
dict 모양은 SPEC 4장의 LLM 답 JSON과 같다. 그래서 러너는 상대가 봇인지 LLM인지
몰라도 같은 방식으로 부를 수 있다.

어부는 "몇 마리 잡고 싶은지"만 말한다. 실제로 몇 마리를 받는지,
20마리 상한을 넘었는지는 여기서 정하지 않는다(판정은 `actions.py`·`rules.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """어부 한 명이 어획 직전에 아는 것. SPEC 3장 "아침 공지" + 내 상태.

    정확한 연못 재고(`stock`)는 넣지 않는다. 어부는 추정치만 안다(SPEC 1장 정보 비대칭).
    """

    day: int
    agent_id: str
    money: int
    stock_estimate: int  # 아침에 공개된 연못 추정치
    yesterday_total_catch: int | None = None  # 1일차에는 전날이 없어 None


def fish_action(amount: int, reason: str) -> dict:
    """SPEC 4장 어획 행동 JSON: {"action": "fish", "amount": ..., "reason": ...}."""
    return {"action": "fish", "amount": amount, "reason": reason}


class BaseAgent(ABC):
    """어부 한 명의 두뇌. 상태(돈·생존)는 `world.Agent`가 들고, 여기는 결정만 한다."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def decide_fish(self, obs: Observation) -> dict:
        """어획 페이즈 행동을 SPEC 4장 모양의 dict로 돌려준다."""
