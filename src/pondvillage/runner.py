"""하루 페이즈 진행, 실험 실행.

러너는 "진행자"다. 직접 판정하지 않고, SPEC 3장 순서대로 어부에게 묻고 규칙 엔진을 부른다.
- 아침 공지: 연못 추정치(시드 고정 무작위)와 어제 총 어획량
- 광장: 설정 talk.enabled일 때만. 매일 시드로 섞은 순서대로 한 명씩 한 마디(모두에게 / 귓속말).
  뒤 사람은 앞에서 나온 말 중 자기가 들을 수 있는 것을 보고 말한다. 봇은 말하지 않는다
- 어획: 어부 답 → `actions.py` 검사. 연못을 확인한 어부에게는 정확한 재고를 보여 주고 한 번 더 묻는다
- 정산: `rules.py` 함수를 순서대로 (배분 → 빼기 → 판매·생활비 → 파산 → 붕괴 → 재생)
- 기억: 정산 뒤 각 어부가 겪은 사실을 기억에 적는다 (`memory.py`, 코드가 적음)
- 일기: 남은 어부가 저녁 일기를 쓰고(LLM만, 봇은 안 씀), 검사 뒤 요약·믿음·신뢰에 반영한다
- 회의: (3단계에서 붙임)

실행 예:
    uv run python -m pondvillage.runner --config configs/base.yaml --days 5 --bot greedy
    → 결과가 runs/<시각>_greedy_s1.db에 저장된다 (--no-log면 저장 안 함)
    uv run python -m pondvillage.runner --llm --days 5                               # qwen2.5:7b × 5
    uv run python -m pondvillage.runner --llm --days 5 --models qwen2.5:7b,gemma3:4b # 섞인 마을
    → runs/<시각>_llm-same_s1.db / runs/<시각>_llm-mixed_s1.db (LLM 호출은 llm_calls 표에)
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

from collections import Counter
from collections.abc import Callable

from pondvillage import llm, memory, rules
from pondvillage.actions import (
    INSPECT, NONE, TO_ALL, TO_SELF, FishDecision, SayDecision, catch_cap, validate_fish_action, validate_say_action,
)
from pondvillage.agents.base import BaseAgent, DiaryContext, Observation, SayContext
from pondvillage.agents.llm_agent import build_llm_agents
from pondvillage.agents.scripted import CooperativeBot, GreedyBot, RandomBot
from pondvillage.logger import (
    STATUS_ABORTED_COST, STATUS_ERROR, RunLogger, format_tokens, make_run_id, make_stamp, open_db,
)
from pondvillage.world import Agent, Config, World, load_config

# 끝난 이유
END_DAYS = "days"  # 정한 일수 도달
END_COLLAPSED = "collapsed"  # 연못 붕괴
END_ALL_LEFT = "all_left"  # 모두 떠남


# ---------------------------------------------------------------- 무작위 (시드 고정)


def make_rng(seed: int, purpose: str) -> random.Random:
    """시드 하나에서 용도별 rng를 갈라 만든다 (지침 규칙 4).

    용도마다 따로 두는 이유: 랜덤 봇을 넣고 빼도 아침 추정치 순서가 흔들리지 않게 하려고.
    문자열 시드는 파이썬 실행마다 같은 결과를 낸다.
    """
    return random.Random(f"{seed}:{purpose}")


def morning_estimate(stock: int, noise: tuple[float, float], rng: random.Random) -> int:
    """아침에 공개할 연못 추정치. SPEC 1장: round(stock * U(0.8, 1.2))."""
    low, high = noise
    return round(stock * rng.uniform(low, high))


# ---------------------------------------------------------------- 하루 결과


@dataclass(frozen=True)
class FishTurn:
    """어부 한 명의 어획 페이즈 기록."""

    agent_id: str
    inspected: bool  # 연못을 확인했는지 (그랬다면 decision은 다시 물은 뒤의 답)
    decision: FishDecision  # 최종 어획 결정
    first: FishDecision | None = None  # 확인했다면 첫 답(inspect_pond). 아니면 None
    exact_stock: int | None = None  # 확인했다면 그때 본 정확한 재고 (기억에 적는다)


@dataclass(frozen=True)
class SayTurn:
    """어부 한 명의 광장 기록 (말한 순서대로 쌓인다)."""

    agent_id: str
    decision: SayDecision


@dataclass
class DayResult:
    """하루 동안 일어난 일. 콘솔 표와 SQLite 로그(`logger.py`)가 이걸 읽는다."""

    day: int
    stock_start: int
    estimate: int
    turns: list[FishTurn]
    requests: dict[str, int]  # 검사 뒤 요청 마릿수
    catches: dict[str, int]  # 실제로 잡은 마릿수
    stock_after_catch: int
    stock_end: int  # 재생 뒤
    collapsed: bool
    left: list[str]  # 오늘 마을을 떠난 어부 id
    money: dict[str, int] = field(default_factory=dict)  # 정산 뒤 돈 (모든 어부)
    alive: dict[str, bool] = field(default_factory=dict)
    diaries: list[memory.DiaryEntry] = field(default_factory=list)  # 오늘 쓴 일기 (봇 마을은 빈 목록)
    says: list[SayTurn] = field(default_factory=list)  # 오늘 광장 (말한 순서. 대화를 끄거나 봇이면 빈 목록)

    @property
    def total_catch(self) -> int:
        return sum(self.catches.values())

    @property
    def messages(self) -> list[memory.PlazaMessage]:
        return plaza_messages(self.says)


def plaza_messages(says: list[SayTurn]) -> list[memory.PlazaMessage]:
    """광장에서 실제로 나온 말 (해석 실패로 말하지 않은 것은 뺀다)."""
    return [memory.PlazaMessage(s.agent_id, s.decision.to, s.decision.message) for s in says if not s.decision.silent]


@dataclass
class RunResult:
    days: list[DayResult]
    end_reason: str


# ---------------------------------------------------------------- 하루 진행


def _names(world: World) -> dict[str, str]:
    return {a.id: a.name for a in world.agents}


def _trust_names(world: World, agent_id: str) -> dict[str, str] | None:
    """질문지에 신뢰 점수를 붙일 남은 이웃 (id → 이름). 대화를 끈 마을은 None (P5 질문지 그대로).

    혼자 말하기 마을도 None: 이웃의 말을 듣지 못하니 신뢰 점수는 근거 없는 숫자다 (대화 끈 마을과 같은 이유).
    """
    if not world.config.talk.enabled or world.config.talk.solo:
        return None
    return {a.id: a.name for a in world.alive_agents() if a.id != agent_id}


def _plaza_phase(
    world: World,
    agents: dict[str, BaseAgent],
    estimate: int,
    yesterday: int | None,
    rng: random.Random,
) -> list[SayTurn]:
    """광장: 시드로 섞은 순서대로 한 마디씩. 각자 내 차례 전까지 나온 말 중 들을 수 있는 것만 본다.

    혼자 말하기 마을(talk.mode = solo, P8): 한 마디씩 쓰지만 받는 사람은 늘 자기 자신(TO_SELF)이라
    아무도 남의 말을 듣지 않는다. 순서도 똑같이 섞는다 (rng 쓰임새를 광장 마을과 맞추려고).
    """
    alive = world.alive_agents()
    order = list(alive)
    rng.shuffle(order)  # 봇 마을에서도 섞어 둔다 (rng 쓰임새가 마을 구성에 따라 흔들리지 않게)
    names = _names(world)
    name_to_id = {a.name: a.id for a in world.agents}
    listeners = {a.id for a in alive}
    solo = world.config.talk.solo
    says: list[SayTurn] = []
    spoken: list[memory.PlazaMessage] = []
    for agent in order:
        ctx = SayContext(
            day=world.day,
            agent_id=agent.id,
            money=agent.money,
            stock_estimate=estimate,
            yesterday_total_catch=yesterday,
            memory=memory.memory_text(agent, world.day - 1, trust_names=_trust_names(world, agent.id)),
            heard=memory.heard_text(spoken, agent.id, names),
            neighbors=tuple(a.name for a in alive if a.id != agent.id),
        )
        raw = agents[agent.id].say(ctx)
        if raw is None:  # 봇
            continue
        decision = validate_say_action(raw, agent.id, name_to_id, listeners, world.config.talk.max_chars)
        if solo and not decision.silent:
            decision = replace(decision, to=TO_SELF)
        says.append(SayTurn(agent.id, decision))
        if not decision.silent:
            spoken.append(memory.PlazaMessage(agent.id, decision.to, decision.message))
    return says


def by_model(agents: list[Agent]) -> list[Agent]:
    """같은 모델끼리 이어서 부르도록 다시 줄 세운다 (모델이 나온 순서대로, 그 안에서는 마을 순서).

    그래픽카드에 모델이 하나씩만 올라가서, 섞인 마을은 모델을 바꿀 때마다 몇 초씩 다시 올린다 (P4: 3~4배 느림).
    어획·일기는 어부들이 서로의 답을 모르고 따로 정하므로 부르는 순서를 바꿔도 결과는 같다 (P8).
    광장은 뒤 사람이 앞 사람 말을 듣기 때문에 묶지 않는다.
    """
    first_seen: dict[str | None, int] = {}
    for a in agents:
        first_seen.setdefault(a.model, len(first_seen))
    return sorted(agents, key=lambda a: first_seen[a.model])  # sorted는 같은 값끼리 원래 순서를 지킨다


def _in_village_order(world: World, turns: list[FishTurn]) -> list[FishTurn]:
    """기록·표는 늘 마을 순서(설정 파일 순서)로 남긴다."""
    order = {a.id: i for i, a in enumerate(world.agents)}
    return sorted(turns, key=lambda t: order[t.agent_id])


def _fish_phase(
    world: World,
    agents: dict[str, BaseAgent],
    estimate: int,
    yesterday: int | None,
    messages: list[memory.PlazaMessage] | None = None,
) -> list[FishTurn]:
    economy = world.config.economy
    names = _names(world)
    turns: list[FishTurn] = []
    for agent in by_model(world.alive_agents()):
        brain = agents[agent.id]
        obs = Observation(
            day=world.day,
            agent_id=agent.id,
            money=agent.money,
            stock_estimate=estimate,
            yesterday_total_catch=yesterday,
            memory=memory.memory_text(agent, world.day - 1, trust_names=_trust_names(world, agent.id)),
            heard=memory.heard_text(messages or [], agent.id, names),
        )
        first = validate_fish_action(brain.decide_fish(obs), catch_cap(economy, inspected=False))
        if first.kind != INSPECT:
            turns.append(FishTurn(agent.id, inspected=False, decision=first))
            continue
        # 연못 확인: 정확한 재고를 보여 주고 한 번 더 묻는다. 그날 상한은 절반.
        again = replace(obs, exact_stock=world.pond.stock)
        second = validate_fish_action(
            brain.decide_fish(again), catch_cap(economy, inspected=True), allow_inspect=False
        )
        turns.append(FishTurn(agent.id, inspected=True, decision=second, first=first, exact_stock=again.exact_stock))
    return _in_village_order(world, turns)


def run_day(
    world: World,
    agents: dict[str, BaseAgent],
    estimate_rng: random.Random,
    yesterday_total: int | None = None,
    talk_rng: random.Random | None = None,
) -> DayResult:
    """`world.day` 하루를 진행하고 결과를 돌려준다. 날짜는 올리지 않는다 (`run`이 올린다).

    talk_rng: 광장 말하기 순서를 섞을 rng. 대화를 켰는데 주지 않으면 설정 시드로 새로 만든다
    (하루만 돌려 보는 테스트용. 여러 날은 `run`이 하나를 이어 쓴다).
    """
    stock_start = world.pond.stock

    # 1. 아침 공지
    estimate = morning_estimate(stock_start, world.config.pond.estimate_noise, estimate_rng)

    # 2. 광장 (대화를 켠 마을만)
    says: list[SayTurn] = []
    if world.config.talk.enabled:
        rng = talk_rng if talk_rng is not None else make_rng(world.config.seed, "talk_order")
        says = _plaza_phase(world, agents, estimate, yesterday_total, rng)
    messages = plaza_messages(says)

    # 3. 어획
    turns = _fish_phase(world, agents, estimate, yesterday_total, messages)
    requests = {t.agent_id: t.decision.amount for t in turns}

    # 4. 정산 (SPEC 3장 순서)
    catches = rules.allocate_catch(requests, world.pond.stock)
    rules.remove_catch(world, catches)
    stock_after_catch = world.pond.stock
    rules.settle_money(world, catches)
    left = rules.check_bankruptcy(world)
    collapsed = rules.check_collapse(world)
    rules.regenerate(world)

    # 5. 기억 적기 (오늘 물어본 어부 전원, 오늘 떠난 어부 포함)
    _remember_day(world, turns, catches, estimate, yesterday_total, left, messages)

    # 6. 일기 (남은 어부만. 붕괴한 날·마지막 날에도 써서 믿음이 매일 기록되게 한다)
    diaries = _diary_phase(world, agents)

    return DayResult(
        day=world.day,
        stock_start=stock_start,
        estimate=estimate,
        turns=turns,
        requests=requests,
        catches=catches,
        stock_after_catch=stock_after_catch,
        stock_end=world.pond.stock,
        collapsed=collapsed,
        left=left,
        money={a.id: a.money for a in world.agents},
        alive={a.id: a.alive for a in world.agents},
        diaries=diaries,
        says=says,
    )


def _remember_day(
    world: World,
    turns: list[FishTurn],
    catches: dict[str, int],
    estimate: int,
    yesterday: int | None,
    left: list[str],
    messages: list[memory.PlazaMessage] | None = None,
) -> None:
    eco = world.config.economy
    names = _names(world)
    left_names = [world.get_agent(i).name for i in left]
    for t in turns:
        agent = world.get_agent(t.agent_id)
        caught = catches.get(t.agent_id, 0)
        text = memory.describe_day(
            world.day, estimate, yesterday,
            kind=t.decision.kind, amount=t.decision.amount, caught=caught,
            exact_stock=t.exact_stock, parse_error=t.decision.parse_error is not None,
            earned=caught * eco.fish_price, cost=eco.daily_cost, money=agent.money,
            left_names=[n for n in left_names if n != agent.name],
        )
        memory.remember(agent, world.day, text)
        plaza = memory.describe_plaza(world.day, messages or [], agent.id, names)
        if plaza is not None:
            memory.remember(agent, world.day, plaza)


def _diary_phase(world: World, agents: dict[str, BaseAgent]) -> list[memory.DiaryEntry]:
    alive = world.alive_agents()
    name_to_id = {a.name: a.id for a in world.agents}
    entries: list[memory.DiaryEntry] = []
    for agent in by_model(alive):
        ctx = DiaryContext(
            day=world.day,
            agent_id=agent.id,
            money=agent.money,
            memory=memory.memory_text(agent, world.day, trust_names=_trust_names(world, agent.id)),
            neighbors=tuple(a.name for a in alive if a.id != agent.id),
        )
        raw = agents[agent.id].write_diary(ctx)
        if raw is None:  # 봇
            continue
        entry = memory.validate_diary(
            raw, agent.id, name_to_id,
            capacity=world.config.pond.capacity, max_catch=world.config.economy.max_catch_per_day,
        )
        memory.apply_diary(agent, world.day, entry)
        entries.append(entry)
    order = {a.id: i for i, a in enumerate(world.agents)}
    return sorted(entries, key=lambda e: order[e.agent_id])


def run(
    world: World,
    agents: dict[str, BaseAgent],
    seed: int,
    days: int | None = None,
    logger: RunLogger | None = None,
    on_day: Callable[[DayResult], None] | None = None,
) -> RunResult:
    """끝날 때까지 하루씩 돌린다. SPEC 8장 종료 조건: 일수 도달, 연못 붕괴, 모두 떠남, 비용 상한.

    logger를 주면 매일 저녁 하루 결과를 기록하고, 끝나면 끝난 이유를 runs.status에 적는다.
    비용 상한을 넘으면 status를 aborted_cost로, 그 밖의 오류(LLM 서버 꺼짐 등)는 error로 적고
    오류는 그대로 올려 보낸다. on_day: 하루가 끝날 때마다 부를 함수 (콘솔 진행 표시용).
    """
    try:
        result = _run_days(world, agents, seed, days, logger, on_day)
    except llm.CostCapExceeded:
        if logger is not None:
            logger.finish(STATUS_ABORTED_COST)
        raise
    except Exception:
        if logger is not None:
            logger.finish(STATUS_ERROR)
        raise
    if logger is not None:
        logger.finish(result.end_reason)
    return result


def _run_days(
    world: World,
    agents: dict[str, BaseAgent],
    seed: int,
    days: int | None,
    logger: RunLogger | None,
    on_day: Callable[[DayResult], None] | None = None,
) -> RunResult:
    days = world.config.days if days is None else days
    estimate_rng = make_rng(seed, "estimate")
    talk_rng = make_rng(seed, "talk_order")
    results: list[DayResult] = []
    yesterday: int | None = None
    while True:
        result = run_day(world, agents, estimate_rng, yesterday, talk_rng)
        results.append(result)
        if logger is not None:
            logger.log_day(result, world)
        if on_day is not None:
            on_day(result)
        if result.collapsed:
            return RunResult(results, END_COLLAPSED)
        if not world.alive_agents():
            return RunResult(results, END_ALL_LEFT)
        if world.day >= days:
            return RunResult(results, END_DAYS)
        yesterday = result.total_catch
        world.day += 1


# ---------------------------------------------------------------- 봇 마을 실행 (콘솔)

BOT_KINDS = ("greedy", "coop", "random")


def build_bots(config: Config, kind: str, seed: int) -> dict[str, BaseAgent]:
    """설정의 어부 5명을 모두 같은 종류의 봇으로 만든다. 랜덤 봇은 어부마다 따로 rng를 갈라 준다."""
    bots: dict[str, BaseAgent] = {}
    for spec in config.agents:
        if kind == "greedy":
            bots[spec.id] = GreedyBot(spec.id)
        elif kind == "coop":
            bots[spec.id] = CooperativeBot(spec.id)
        elif kind == "random":
            bots[spec.id] = RandomBot(spec.id, make_rng(seed, f"bot:{spec.id}"))
        else:
            raise ValueError(f"모르는 봇 종류: {kind} (가능: {', '.join(BOT_KINDS)})")
    return bots


END_LABELS = {
    END_DAYS: "정한 날짜까지 버팀",
    END_COLLAPSED: "연못 붕괴",
    END_ALL_LEFT: "어부가 모두 떠남",
}


def format_table(world: World, result: RunResult) -> str:
    """날짜별 표: 아침 재고, 추정치, 총 어획, 저녁 재고(재생 뒤), 어부별 돈."""
    names = [a.name for a in world.agents]
    header = ["날", "아침재고", "추정치", "총어획", "저녁재고"] + names
    rows = [header]
    for d in result.days:
        money = [str(d.money[a.id]) if d.alive[a.id] else f"떠남({d.money[a.id]})" for a in world.agents]
        stock_end = f"{d.stock_end} 붕괴" if d.collapsed else str(d.stock_end)
        rows.append([str(d.day), str(d.stock_start), str(d.estimate), str(d.total_catch), stock_end] + money)
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(len(header))]
    lines = ["".join(cell.rjust(w) for cell, w in zip(r, widths)) for r in rows]
    lines.append("")
    lines.append(f"끝난 이유: {END_LABELS[result.end_reason]} ({len(result.days)}일 진행)")
    return "\n".join(lines)


# ---------------------------------------------------------------- 기준선 (봇 종류 × 시드 비교)


@dataclass(frozen=True)
class RunSummary:
    """실행 한 번을 한 줄로 줄인 것. 기준선 비교표가 쓴다.

    DB에는 끝난 이유만 runs.status로 남는다. 나머지는 days·agent_days에서 다시 계산할 수 있다.
    """

    bot: str
    seed: int
    end_reason: str
    days: int  # 진행한 날 수
    stock_end: int  # 마지막 날 저녁 재고
    total_catch: int  # 마을 전체 누적 어획
    left: int  # 떠난 어부 수
    avg_money: float  # 마지막 날 평균 돈 (떠난 어부 포함)


def summarize(bot: str, seed: int, result: RunResult) -> RunSummary:
    last = result.days[-1]
    return RunSummary(
        bot=bot,
        seed=seed,
        end_reason=result.end_reason,
        days=len(result.days),
        stock_end=last.stock_end,
        total_catch=sum(d.total_catch for d in result.days),
        left=sum(1 for alive in last.alive.values() if not alive),
        avg_money=sum(last.money.values()) / len(last.money),
    )


BOT_MODEL = "scripted"  # runs.model 칸: 규칙 기반 봇 (LLM 아님)


def run_baseline(
    config: Config,
    seeds: list[int],
    kinds: tuple[str, ...] = BOT_KINDS,
    days: int | None = None,
    conn: sqlite3.Connection | None = None,
    config_name: str = "base",
    stamp: str | None = None,
) -> list[RunSummary]:
    """봇 종류마다, 시드마다 새 마을을 만들어 끝까지 돌리고 한 줄 요약을 모은다.

    conn을 주면 모든 실행을 그 DB 파일 하나에 기록한다 (run_id로 구분).
    """
    stamp = stamp or make_stamp()
    out: list[RunSummary] = []
    for kind in kinds:
        for seed in seeds:
            world = World.from_config(config)
            logger = None
            if conn is not None:
                logger = RunLogger.start(
                    conn, make_run_id(stamp, kind, seed),
                    config_name=config_name, condition=kind, seed=seed, model=BOT_MODEL,
                )
            result = run(world, build_bots(config, kind, seed), seed, days=days, logger=logger)
            out.append(summarize(kind, seed, result))
    return out


BOT_LABELS = {"greedy": "탐욕", "coop": "협력", "random": "랜덤"}


def format_baseline(rows: list[RunSummary]) -> str:
    """시드별 한 줄 + 봇 종류별 요약."""
    header = ["봇", "시드", "끝난 이유", "일수", "저녁재고", "총어획", "떠남", "평균돈"]
    table = [header]
    for r in rows:
        table.append([
            BOT_LABELS.get(r.bot, r.bot), str(r.seed), END_LABELS[r.end_reason], str(r.days),
            str(r.stock_end), str(r.total_catch), str(r.left), f"{r.avg_money:.1f}",
        ])
    widths = [max(len(t[i]) for t in table) + 2 for i in range(len(header))]
    lines = ["".join(cell.rjust(w) for cell, w in zip(t, widths)) for t in table]
    lines.append("")
    lines.append("요약")
    for kind in dict.fromkeys(r.bot for r in rows):
        group = [r for r in rows if r.bot == kind]
        n = len(group)
        counts = {reason: sum(1 for r in group if r.end_reason == reason) for reason in END_LABELS}
        reasons = ", ".join(f"{END_LABELS[k]} {v}" for k, v in counts.items() if v)
        avg_days = sum(r.days for r in group) / n
        lines.append(f"  {BOT_LABELS.get(kind, kind)}: {n}번 중 {reasons} · 평균 {avg_days:.1f}일")
    return "\n".join(lines)


# ---------------------------------------------------------------- LLM 마을 (모델 마을, EXPERIMENT 1단계)


def assign_models(config: Config, override: list[str] | None = None) -> dict[str, str]:
    """어부 id → 모델. override가 있으면 어부 순서대로 번갈아 준다 (예: q, g → q g q g q).

    없으면 설정의 어부별 model, 그것도 없으면 기본 모델. 정식 마을 구성 파일은 P8(exp1_*.yaml).
    """
    if override:
        return {spec.id: override[i % len(override)] for i, spec in enumerate(config.agents)}
    return {spec.id: config.model_for(spec) for spec in config.agents}


def village_label(models: dict[str, str]) -> str:
    """runs.model에 적을 마을 구성 이름. 같은 모델이면 그 이름, 섞이면 예: 3qwen2.5:7b+2gemma3:4b."""
    counts = Counter(models.values())
    if len(counts) == 1:
        return next(iter(counts))
    return "+".join(f"{n}{m}" for m, n in counts.items())


def village_condition(models: dict[str, str]) -> str:
    return "llm-same" if len(set(models.values())) == 1 else "llm-mixed"


def format_requests(world: World, result: RunResult) -> str:
    """날짜별 어부 답: 요청 마릿수, 연못을 확인했으면 "확인→n", 해석 실패면 "실패"."""
    header = ["날"] + [f"{a.name}({a.model})" if a.model else a.name for a in world.agents]
    rows = [header]
    for d in result.days:
        by_id = {t.agent_id: t for t in d.turns}
        cells = []
        for a in world.agents:
            t = by_id.get(a.id)
            if t is None:
                cells.append("-")
            elif t.decision.parse_error:
                cells.append("실패")
            elif t.inspected:
                cells.append(f"확인→{t.decision.amount}" if t.decision.kind != NONE else "확인→0")
            else:
                cells.append(str(t.decision.amount) + ("*" if t.decision.clamped else ""))
        rows.append([str(d.day)] + cells)
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(len(header))]
    lines = ["".join(cell.rjust(w) for cell, w in zip(r, widths)) for r in rows]
    lines.append("(*: 20마리 상한으로 잘림)")
    return "\n".join(lines)


def format_plaza(world: World, day: DayResult) -> str:
    """하루 광장을 말한 순서대로. 예: 순자 → 모두: "..." / 영철 → 미란(귓속말): "..." """
    names = _names(world)
    lines = []
    for s in day.says:
        d = s.decision
        if d.silent:
            lines.append(f"  {names[s.agent_id]}: (답을 읽지 못해 말하지 않음)")
            continue
        to = "모두" if d.to == TO_ALL else f"{names[d.to]}(귓속말)"
        lines.append(f'  {names[s.agent_id]} → {to}: "{d.message}"' + (" *" if d.clamped else ""))
    return "\n".join(lines) if lines else "  (말한 사람 없음)"


def format_model_summary(world: World, result: RunResult) -> str:
    """모델별 한 줄: 어부 1명 하루 평균 요청 마릿수와 연못 확인 횟수 (analyze.py와 같은 정의).

    요청은 검사 뒤 최종 요청이다. 확인 뒤 안 잡음·해석 실패도 0마리로 평균에 넣는다.
    """
    model_of = {a.id: a.model or "?" for a in world.agents}
    stats: dict[str, list[int]] = {}  # 모델 → [요청 합, 어부-일 수, 확인 횟수]
    for d in result.days:
        for t in d.turns:
            s = stats.setdefault(model_of[t.agent_id], [0, 0, 0])
            s[0] += d.requests.get(t.agent_id, 0)
            s[1] += 1
            s[2] += int(t.inspected)
    parts = [
        f"{m} 평균 요청 {total / n:.1f}마리 (어부-일 {n}) · 연못 확인 {inspected}번"
        for m, (total, n, inspected) in stats.items() if n
    ]
    return "모델별: " + " / ".join(parts) if parts else "모델별: (기록 없음)"


def talk_label(config: Config) -> str:
    if not config.talk.enabled:
        return "끔"
    return "혼자 말하기" if config.talk.solo else "켬"


@dataclass
class VillageRun:
    """LLM 마을 한 번 실행의 결과. 멈췄으면 result는 None이고 error에 이유."""

    run_id: str
    db_path: Path | None
    world: World
    calls: list[llm.CallRecord]
    result: RunResult | None = None
    error: str | None = None


def run_llm_village(
    config: Config,
    models: dict[str, str],
    seed: int,
    *,
    run_id: str,
    condition: str,
    config_name: str,
    runs_dir: Path,
    days: int | None = None,
    log: bool = True,
    on_day: Callable[[DayResult], None] | None = None,
) -> VillageRun:
    """LLM 마을 하나를 만들어 끝까지 돌린다. 러너(--llm)와 실험 러너(experiment.py)가 같이 쓴다.

    Ollama 점검은 부르는 쪽이 먼저 한다 (실험은 마을 여러 개 앞에서 한 번만).
    """
    assert config.llm is not None
    world = World.from_config(config)
    for agent in world.agents:
        agent.model = models[agent.id]
    db_path = runs_dir / f"{run_id}.db" if log else None
    conn = open_db(db_path) if db_path is not None else None

    calls: list[llm.CallRecord] = []
    logger = None
    if conn is not None:
        logger = RunLogger.start(
            conn, run_id, config_name=config_name, condition=condition, seed=seed, model=village_label(models)
        )

    def record(rec: llm.CallRecord) -> None:
        calls.append(rec)
        if logger is not None:
            logger.log_llm_call(rec)

    caller = llm.LLMCaller(
        config.llm.base_url, config.llm.temperature, meter=llm.CostMeter(config.llm.cost_cap_usd), record=record
    )
    agents = build_llm_agents(config, caller, models)
    out = VillageRun(run_id, db_path, world, calls)
    try:
        out.result = run(world, agents, seed, days=days, logger=logger, on_day=on_day)
    except (llm.LLMUnavailable, llm.CostCapExceeded) as e:
        out.error = str(e)
    finally:
        if conn is not None:
            conn.close()
    return out


def _main_llm(args: argparse.Namespace, config: Config, config_name: str, stamp: str, runs_dir: Path) -> int:
    if config.llm is None:
        print("설정 파일에 llm 항목이 없습니다.")
        return 1
    override = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    models = assign_models(config, override)

    # 출발 전 점검: 꺼진 서버에 120초씩 기다리지 않게 (WSL이 쉬면 Ollama도 꺼진다)
    try:
        llm.check_ollama(config.llm.base_url, list(models.values()))
    except (llm.LLMUnavailable, llm.PaidCallNotAllowed) as e:
        print(f"[시작 못 함] {e}")
        return 1

    seed = config.seed if args.seed is None else args.seed
    condition = village_condition(models)
    print(
        f"LLM 마을: {village_label(models)} · 시드 {seed} · 성격 문구 {'켬' if config.llm.persona_enabled else '끔'}"
        f" · 광장 대화 {talk_label(config)}"
    )
    vr = run_llm_village(
        config, models, seed, run_id=make_run_id(stamp, condition, seed), condition=condition,
        config_name=config_name, runs_dir=runs_dir, days=args.days, log=not args.no_log, on_day=print_progress,
    )
    world, calls, db_path = vr.world, vr.calls, vr.db_path
    if vr.result is None:
        print(f"[멈춤] {vr.error}")
        if db_path is not None:
            print(f"그때까지의 기록: {db_path}")
        return 1
    result = vr.result

    print(format_table(world, result))
    print("\n어부 답 (요청 마릿수)")
    print(format_requests(world, result))
    print(format_model_summary(world, result))
    if config.talk.enabled:
        print("\n1일째 광장 (말한 순서, *: 잘렸거나 받는 사람을 몰라 모두에게로 바꿈)")
        print(format_plaza(world, result.days[0]))
    failed = sum(1 for c in calls if c.parse_error)
    avg_ms = sum(c.latency_ms for c in calls) / len(calls) if calls else 0
    print(
        f"\nLLM 호출 {len(calls)}번 · 해석 실패 {failed}번 · 평균 {avg_ms / 1000:.1f}초"
        f" · 비용 {sum(c.cost_usd for c in calls):.4f} USD"
    )
    first_asks = [(c.day, c.phase, c.prompt_tokens) for c in calls if '"role": "assistant"' not in c.prompt]
    print("\n질문지 길이 (프롬프트 토큰, 첫 질문만)")
    print(format_tokens(first_asks))
    if db_path is not None:
        print(f"저장: {db_path}")
    return 0


def print_progress(d: DayResult) -> None:
    print(f"  {d.day}일째 끝: 총어획 {d.total_catch}, 저녁 재고 {d.stock_end}{' (붕괴)' if d.collapsed else ''}",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="봇 마을을 돌려 날짜별 표를 찍는다 (LLM 없음, 비용 0원).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--days", type=int, default=None, help="진행할 날 수 (기본: 설정 값)")
    parser.add_argument("--bot", choices=BOT_KINDS, default="coop", help="5명 모두 이 봇으로")
    parser.add_argument("--seed", type=int, default=None, help="시드 (기본: 설정 값)")
    parser.add_argument("--baseline", action="store_true", help="봇 3종 × 시드 1~N 비교표")
    parser.add_argument("--seeds", type=int, default=10, help="--baseline에서 돌릴 시드 수 (기본 10)")
    parser.add_argument("--runs-dir", default="runs", help="DB 파일을 둘 폴더 (기본 runs)")
    parser.add_argument("--no-log", action="store_true", help="DB 파일에 저장하지 않는다")
    parser.add_argument("--llm", action="store_true", help="봇 대신 LLM 어부 (로컬 Ollama)")
    parser.add_argument(
        "--models", default=None,
        help="--llm에서 어부에게 번갈아 줄 모델, 쉼표로 (예: qwen2.5:7b,gemma3:4b). 없으면 설정 값",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config_name = Path(args.config).stem
    stamp = make_stamp()
    runs_dir = Path(args.runs_dir)

    if args.llm:
        return _main_llm(args, config, config_name, stamp, runs_dir)

    if args.baseline:
        db_path = runs_dir / f"{stamp}_baseline.db"
        conn = None if args.no_log else open_db(db_path)
        try:
            rows = run_baseline(
                config, list(range(1, args.seeds + 1)), days=args.days,
                conn=conn, config_name=config_name, stamp=stamp,
            )
        finally:
            if conn is not None:
                conn.close()
        print(f"기준선: 봇 {len(BOT_KINDS)}종 × 시드 1~{args.seeds}")
        print(format_baseline(rows))
        if conn is not None:
            print(f"\n저장: {db_path}")
        return 0

    seed = config.seed if args.seed is None else args.seed
    world = World.from_config(config)
    run_id = make_run_id(stamp, args.bot, seed)
    db_path = runs_dir / f"{run_id}.db"
    conn = None if args.no_log else open_db(db_path)
    try:
        logger = None
        if conn is not None:
            logger = RunLogger.start(
                conn, run_id, config_name=config_name, condition=args.bot, seed=seed, model=BOT_MODEL
            )
        result = run(world, build_bots(config, args.bot, seed), seed, days=args.days, logger=logger)
    finally:
        if conn is not None:
            conn.close()
    print(f"봇: {args.bot} · 시드: {seed}")
    print(format_table(world, result))
    if conn is not None:
        print(f"\n저장: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
