"""월드 상태: 연못, 에이전트, 날짜.

이 모듈은 "그릇"만 정의한다. 어획 배분·재생·정산·파산 같은 계산은
전부 `rules.py`가 하고, 여기에는 판정 로직을 넣지 않는다.

두 종류로 나뉜다.
- 설정(Config): `configs/*.yaml`에서 읽은 숫자. 실행 중에는 바뀌지 않는다.
- 상태(World): 날짜·연못 재고·어부의 돈처럼 매일 바뀌는 값.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------- 설정 (불변)


@dataclass(frozen=True)
class PondConfig:
    """연못 숫자. SPEC 1장 "연못"과 같아야 한다."""

    capacity: int
    initial_stock: int
    regen_multiplier: int
    collapse_below: int
    estimate_noise: tuple[float, float]  # 아침 추정치 오차 범위 (낮은 값, 높은 값)


@dataclass(frozen=True)
class EconomyConfig:
    """경제 숫자. SPEC 1장 "경제"와 같아야 한다."""

    start_money: int
    fish_price: int
    daily_cost: int
    max_catch_per_day: int
    inspect_catch_factor: float
    bankrupt_after_days: int


TALK_PLAZA = "plaza"
TALK_SOLO = "solo"
TALK_MODES = (TALK_PLAZA, TALK_SOLO)


@dataclass(frozen=True)
class TalkConfig:
    """광장 대화 설정 (SPEC 3장 광장). 끄면 광장 단계를 통째로 건너뛴다.

    설정 파일에 talk 칸이 없으면 끈 것으로 본다 (P5까지의 설정 파일이 그대로 돌게).
    """

    enabled: bool = False
    max_chars: int = 200  # 한 마디 최대 글자 수. 넘으면 코드가 자른다 (2026-11-15 사용자 승인)
    # "plaza": 광장에서 서로 듣는다 (P6). "solo": 한 마디는 쓰지만 자기만 보고 아무도 못 듣는다 (P8 비교용.
    # 대화 효과가 "남의 말을 들어서"인지 "잡기 전에 한 번 더 생각해서"인지 가르려고. 2026-11-15 사용자 승인)
    mode: str = TALK_PLAZA

    def __post_init__(self) -> None:
        if self.mode not in TALK_MODES:
            raise ConfigError(f"talk.mode는 {', '.join(TALK_MODES)} 중 하나여야 합니다: {self.mode}")

    @property
    def solo(self) -> bool:
        """혼자 말하기 마을인지 (대화를 켰고 mode가 solo)."""
        return self.enabled and self.mode == TALK_SOLO


@dataclass(frozen=True)
class AgentSpec:
    """어부 한 명의 고정 소개. 실험 내내 바뀌지 않는다."""

    id: str
    name: str
    persona: str
    model: str | None = None  # 이 어부를 움직이는 LLM. 비우면 llm.model (SPEC 2장)


@dataclass(frozen=True)
class LLMConfig:
    """LLM 설정. 숫자 규칙이 아니라 "누구에게 어떻게 묻는지"라서 SPEC 규칙과는 별개다."""

    base_url: str
    model: str  # 기본 모델 (어부 칸에 model이 없으면 이것)
    temperature: float = 0.7
    persona_enabled: bool = False  # False면 이름만 준다 (1단계 모델 마을, SPEC 2장)
    cost_cap_usd: float | None = None  # 실행 1회 비용 상한. 넘으면 즉시 멈춘다


@dataclass(frozen=True)
class Config:
    """설정 파일 하나를 통째로 담은 것."""

    pond: PondConfig
    economy: EconomyConfig
    agents: tuple[AgentSpec, ...]
    days: int
    seed: int
    llm: LLMConfig | None = None  # 봇만 돌릴 때는 없어도 된다
    talk: TalkConfig = field(default_factory=TalkConfig)  # 광장 대화 (칸이 없으면 끔)

    def model_for(self, agent: AgentSpec) -> str:
        """이 어부를 움직일 모델: 어부 칸의 model, 없으면 기본 모델."""
        if self.llm is None:
            raise ConfigError("설정 파일에 llm 항목이 없습니다")
        return agent.model or self.llm.model


class ConfigError(ValueError):
    """설정 파일에 빠지거나 잘못된 항목이 있을 때."""


def load_config(path: str | Path) -> Config:
    """YAML 설정 파일을 읽어 Config로 만든다. 항목이 빠졌으면 ConfigError."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> Config:
    """이미 읽어 둔 dict를 Config로 만든다 (테스트에서 파일 없이 쓰기 좋다)."""
    try:
        pond_raw = dict(raw["pond"])
        pond_raw["estimate_noise"] = tuple(pond_raw["estimate_noise"])
        config = Config(
            pond=PondConfig(**pond_raw),
            economy=EconomyConfig(**raw["economy"]),
            agents=tuple(AgentSpec(**a) for a in raw["agents"]),
            days=raw["run"]["days"],
            seed=raw["run"]["seed"],
            llm=LLMConfig(**raw["llm"]) if raw.get("llm") else None,
            talk=TalkConfig(**raw["talk"]) if raw.get("talk") else TalkConfig(),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"설정 파일 항목이 빠졌거나 이름이 틀렸습니다: {e}") from e

    ids = [a.id for a in config.agents]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"어부 id가 겹칩니다: {ids}")
    if not ids:
        raise ConfigError("어부가 한 명도 없습니다")
    return config


# ---------------------------------------------------------------- 상태 (매일 바뀜)


@dataclass
class Pond:
    """연못의 현재 상태."""

    capacity: int
    stock: int
    collapsed: bool = False  # 한 번 붕괴하면 재생하지 않는다 (판정은 rules.py)


@dataclass
class MemoryItem:
    """기억 한 조각: 그날 본 공지, 들은 말, 내 행동과 결과. (저장·요약은 memory.py)"""

    day: int
    text: str
    kind: str = "event"  # "event": 코드가 적은 사실, "diary": 저녁 일기 요약 (memory.py)


@dataclass
class Beliefs:
    """어부가 저녁 일기에서 밝힌 생각. SPEC 2장."""

    estimated_stock: int  # 연못에 몇 마리 있다고 생각하는지
    confidence: float  # 0~1
    sustainable_guess: int  # 1인당 몇 마리까지 괜찮다고 생각하는지
    heard_rumor: bool = False  # (2단계) 소문을 들었는지
    believes_rumor: float = 0.0  # (2단계) 소문을 얼마나 믿는지 0~1


@dataclass
class Agent:
    """어부 한 명의 현재 상태. SPEC 2장."""

    id: str
    name: str
    persona: str
    money: int
    alive: bool = True
    debt_days: int = 0  # 돈 < 0 연속 일수
    memory: list[MemoryItem] = field(default_factory=list)
    beliefs: Beliefs | None = None  # 첫 일기를 쓰기 전에는 없다
    trust: dict[str, float] = field(default_factory=dict)  # 다른 어부 id -> 0~1
    model: str | None = None  # 이 어부를 움직이는 LLM (봇 마을이면 설정에 llm이 없을 때 None)


@dataclass
class World:
    """마을 전체: 날짜, 연못, 어부들, 그리고 규칙 엔진이 참고할 설정."""

    config: Config
    pond: Pond
    agents: list[Agent]
    day: int = 1

    @classmethod
    def from_config(cls, config: Config) -> World:
        """설정대로 1일차 아침의 마을을 만든다."""
        pond = Pond(capacity=config.pond.capacity, stock=config.pond.initial_stock)
        agents = [
            Agent(
                id=a.id, name=a.name, persona=a.persona, money=config.economy.start_money,
                model=config.model_for(a) if config.llm else a.model,
            )
            for a in config.agents
        ]
        return cls(config=config, pond=pond, agents=agents)

    def alive_agents(self) -> list[Agent]:
        """아직 마을에 남아 있는 어부들 (설정 파일 순서 그대로)."""
        return [a for a in self.agents if a.alive]

    def get_agent(self, agent_id: str) -> Agent:
        """id로 어부를 찾는다. 없으면 KeyError."""
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(agent_id)
