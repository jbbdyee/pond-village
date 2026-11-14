"""규칙 기반 봇 (탐욕/협력/랜덤): 검증·기준선용.

LLM 없이 정해진 규칙대로 어획량을 고른다. 두 가지에 쓴다.
- 규칙 엔진 검증: 전원 탐욕이면 붕괴, 전원 협력이면 30일 유지되는지
- 비교 기준선: LLM 마을의 결과를 "양 끝"과 나란히 놓고 본다

기본 숫자 20·10은 docs/EXPERIMENT.md "기준선"에 적힌 값이다.
"""

from __future__ import annotations

import random

from pondvillage.agents.base import BaseAgent, Observation, fish_action


def _check_amount(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name}는 0 이상이어야 합니다: {value}")


class GreedyBot(BaseAgent):
    """매일 최대한(기본 20마리) 잡으려는 봇."""

    def __init__(self, agent_id: str, amount: int = 20) -> None:
        super().__init__(agent_id)
        _check_amount("amount", amount)
        self.amount = amount

    def decide_fish(self, obs: Observation) -> dict:
        return fish_action(self.amount, "탐욕 봇: 매일 최대한 잡는다")


class CooperativeBot(BaseAgent):
    """매일 지속 가능한 몫(기본 10마리 = 50마리 ÷ 5명)만 잡는 봇."""

    def __init__(self, agent_id: str, amount: int = 10) -> None:
        super().__init__(agent_id)
        _check_amount("amount", amount)
        self.amount = amount

    def decide_fish(self, obs: Observation) -> dict:
        return fish_action(self.amount, "협력 봇: 매일 지속 가능한 몫만 잡는다")


class RandomBot(BaseAgent):
    """매일 low~high(기본 0~20) 중 하나를 고르는 봇.

    무작위는 밖에서 받은 `random.Random`만 쓴다(지침 규칙 4). 같은 시드면 같은 순서.
    """

    def __init__(
        self, agent_id: str, rng: random.Random, low: int = 0, high: int = 20
    ) -> None:
        super().__init__(agent_id)
        _check_amount("low", low)
        if high < low:
            raise ValueError(f"high({high})가 low({low})보다 작습니다")
        self.rng = rng
        self.low = low
        self.high = high

    def decide_fish(self, obs: Observation) -> dict:
        amount = self.rng.randint(self.low, self.high)
        return fish_action(amount, f"랜덤 봇: {self.low}~{self.high} 중 무작위")
