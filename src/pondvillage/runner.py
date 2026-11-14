"""하루 페이즈 진행, 실험 실행.

러너는 "진행자"다. 직접 판정하지 않고, SPEC 3장 순서대로 어부에게 묻고 규칙 엔진을 부른다.
- 아침 공지: 연못 추정치(시드 고정 무작위)와 어제 총 어획량
- 광장: (P6에서 붙임)
- 어획: 어부 답 → `actions.py` 검사. 연못을 확인한 어부에게는 정확한 재고를 보여 주고 한 번 더 묻는다
- 정산: `rules.py` 함수를 순서대로 (배분 → 빼기 → 판매·생활비 → 파산 → 붕괴 → 재생)
- 회의·일기: (P5·P8에서 붙임)

실행 예:
    uv run python -m pondvillage.runner --config configs/base.yaml --days 5 --bot greedy
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace

from pondvillage import rules
from pondvillage.actions import INSPECT, FishDecision, catch_cap, validate_fish_action
from pondvillage.agents.base import BaseAgent, Observation
from pondvillage.agents.scripted import CooperativeBot, GreedyBot, RandomBot
from pondvillage.world import Config, World, load_config

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


@dataclass
class DayResult:
    """하루 동안 일어난 일. 콘솔 표와 (다음 항목) SQLite 로그가 이걸 읽는다."""

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

    @property
    def total_catch(self) -> int:
        return sum(self.catches.values())


@dataclass
class RunResult:
    days: list[DayResult]
    end_reason: str


# ---------------------------------------------------------------- 하루 진행


def _fish_phase(world: World, agents: dict[str, BaseAgent], estimate: int, yesterday: int | None) -> list[FishTurn]:
    economy = world.config.economy
    turns: list[FishTurn] = []
    for agent in world.alive_agents():
        brain = agents[agent.id]
        obs = Observation(
            day=world.day,
            agent_id=agent.id,
            money=agent.money,
            stock_estimate=estimate,
            yesterday_total_catch=yesterday,
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
        turns.append(FishTurn(agent.id, inspected=True, decision=second, first=first))
    return turns


def run_day(
    world: World,
    agents: dict[str, BaseAgent],
    estimate_rng: random.Random,
    yesterday_total: int | None = None,
) -> DayResult:
    """`world.day` 하루를 진행하고 결과를 돌려준다. 날짜는 올리지 않는다 (`run`이 올린다)."""
    stock_start = world.pond.stock

    # 1. 아침 공지
    estimate = morning_estimate(stock_start, world.config.pond.estimate_noise, estimate_rng)

    # 3. 어획
    turns = _fish_phase(world, agents, estimate, yesterday_total)
    requests = {t.agent_id: t.decision.amount for t in turns}

    # 4. 정산 (SPEC 3장 순서)
    catches = rules.allocate_catch(requests, world.pond.stock)
    rules.remove_catch(world, catches)
    stock_after_catch = world.pond.stock
    rules.settle_money(world, catches)
    left = rules.check_bankruptcy(world)
    collapsed = rules.check_collapse(world)
    rules.regenerate(world)

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
    )


def run(world: World, agents: dict[str, BaseAgent], seed: int, days: int | None = None) -> RunResult:
    """끝날 때까지 하루씩 돌린다. SPEC 8장 종료 조건: 일수 도달, 연못 붕괴, 모두 떠남.

    (비용 상한은 LLM 어부가 생기는 P4에서 붙인다.)
    """
    days = world.config.days if days is None else days
    estimate_rng = make_rng(seed, "estimate")
    results: list[DayResult] = []
    yesterday: int | None = None
    while True:
        result = run_day(world, agents, estimate_rng, yesterday)
        results.append(result)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="봇 마을을 돌려 날짜별 표를 찍는다 (LLM 없음, 비용 0원).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--days", type=int, default=None, help="진행할 날 수 (기본: 설정 값)")
    parser.add_argument("--bot", choices=BOT_KINDS, default="coop", help="5명 모두 이 봇으로")
    parser.add_argument("--seed", type=int, default=None, help="시드 (기본: 설정 값)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    world = World.from_config(config)
    result = run(world, build_bots(config, args.bot, seed), seed, days=args.days)
    print(f"봇: {args.bot} · 시드: {seed}")
    print(format_table(world, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
