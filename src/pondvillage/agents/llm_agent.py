"""LLM 어부: 아침에 본 것을 질문지로 만들어 자기 모델에게 묻고, 답(JSON)을 행동 dict로 돌려준다.

봇과 같은 약속(`BaseAgent.decide_fish`)을 지키므로 러너는 상대가 봇인지 LLM인지 모른다.
- 어부마다 `model`이 다르다 (모델 마을, EXPERIMENT 1단계). 창구(`LLMCaller`)는 마을이 함께 쓴다.
- 질문지 문구는 `prompts/`의 글 파일에 있다. 숫자는 설정에서 채운다 (SPEC과 어긋나지 않게).
- persona_enabled가 False면 이름만 준다. 성격 문구가 섞이면 차이가 모델 탓인지 가릴 수 없어서다.
- 넣지 않는 것: 지속 가능한 양(스스로 알아내는지가 관찰 대상), 전체 일수(끝을 알면 몰아 잡기 쉬움),
  다른 어부의 모델 이름.

어부는 "몇 마리 잡고 싶은지"만 말한다. 판정은 `actions.py`·`rules.py`.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

from pondvillage.actions import PARSE_ERROR_KEY
from pondvillage.agents.base import BaseAgent, Observation
from pondvillage.llm import LLMCaller
from pondvillage.logger import PHASE_FISH, PHASE_FISH_AFTER_INSPECT
from pondvillage.world import Config

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def fish_system_prompt(config: Config) -> str:
    """규칙 설명. 숫자는 전부 설정 파일에서 온다."""
    pond, eco = config.pond, config.economy
    return load_prompt("fish_system").format(
        n_agents=len(config.agents),
        capacity=pond.capacity,
        regen=pond.regen_multiplier,
        collapse=pond.collapse_below,
        price=eco.fish_price,
        cost=eco.daily_cost,
        bankrupt=eco.bankrupt_after_days,
        cap=eco.max_catch_per_day,
        inspect_cap=math.floor(eco.max_catch_per_day * eco.inspect_catch_factor),
    )


def fish_user_prompt(name: str, persona: str | None, obs: Observation, inspect_cap: int) -> str:
    """오늘 아침 상황. persona가 None이면 이름만."""
    if obs.yesterday_total_catch is None:
        yesterday = "오늘이 첫날이라 어제 기록은 없다."
    else:
        yesterday = f"어제 마을 전체 어획 {obs.yesterday_total_catch}마리."
    inspect = ""
    if obs.exact_stock is not None:
        inspect = (
            f"너는 오늘 연못을 직접 확인했다. 정확한 재고는 {obs.exact_stock}마리다. "
            f"오늘은 최대 {inspect_cap}마리까지 잡을 수 있다. 이제 fish로 답하라.\n"
        )
    return load_prompt("fish_user").format(
        name=name,
        persona_line=f" {persona}" if persona else "",
        day=obs.day,
        estimate=obs.stock_estimate,
        yesterday_line=yesterday,
        money=obs.money,
        inspect_line=inspect,
    )


class LLMAgent(BaseAgent):
    """어부 한 명. 자기 이름·모델로 공용 창구에 묻는다."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        model: str,
        config: Config,
        caller: LLMCaller,
        persona: str | None = None,
    ) -> None:
        super().__init__(agent_id)
        self.name = name
        self.model = model
        self.caller = caller
        self.persona = persona  # None이면 이름만 (persona_enabled: false)
        self.system = fish_system_prompt(config)
        eco = config.economy
        self.inspect_cap = math.floor(eco.max_catch_per_day * eco.inspect_catch_factor)

    def messages_for(self, obs: Observation) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": fish_user_prompt(self.name, self.persona, obs, self.inspect_cap)},
        ]

    def decide_fish(self, obs: Observation) -> dict:
        phase = PHASE_FISH if obs.exact_stock is None else PHASE_FISH_AFTER_INSPECT
        reply = self.caller.ask_json(
            self.messages_for(obs), model=self.model, day=obs.day, agent_id=self.agent_id, phase=phase
        )
        if reply.data is None:
            return {PARSE_ERROR_KEY: reply.parse_error}
        return reply.data


def build_llm_agents(config: Config, caller: LLMCaller, models: dict[str, str]) -> dict[str, BaseAgent]:
    """설정의 어부 전원을 LLM 어부로. models: 어부 id → 모델 이름."""
    assert config.llm is not None
    persona_on = config.llm.persona_enabled
    return {
        spec.id: LLMAgent(
            spec.id, spec.name, models[spec.id], config, caller,
            persona=spec.persona if persona_on else None,
        )
        for spec in config.agents
    }
