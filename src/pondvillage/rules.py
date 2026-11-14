"""규칙 엔진: 어획 배분, 재생, 정산, 사망 판정 (LLM 없음).

LLM도, 무작위도 쓰지 않는다. 같은 입력이면 언제나 같은 결과가 나온다.
- 계산만 하는 함수(`allocate_catch`)는 입력을 고치지 않고 답만 돌려준다.
- 정산 함수(`settle_money`, `check_bankruptcy`, `remove_catch`, `check_collapse`, `regenerate`)는 `World`의 어부·연못 상태를 직접 고친다.
"""

from __future__ import annotations

from pondvillage.world import World


def allocate_catch(requests: dict[str, int], stock: int) -> dict[str, int]:
    """어부들이 요청한 마릿수를 받아, 실제로 잡히는 마릿수를 돌려준다. SPEC 3장.

    - 요청 합계 ≤ 재고: 요청대로 준다.
    - 요청 합계 > 재고: 요청 비율로 나누고 소수점은 내린다.
      내림 때문에 남는 물고기는 아무에게도 주지 않는다 (연못에 그대로 남는다).

    requests: 어부 id -> 요청 마릿수. 상한(20마리, 연못을 확인한 날은 10마리)은
              이미 `actions.py`에서 잘라 낸 값이라고 본다. 아무것도 안 한 어부는 0으로 넣는다.
    stock:    지금 연못에 있는 물고기 수.

    연못 재고는 여기서 줄이지 않는다. 돌려준 값의 합만큼 빼는 것은 정산 단계의 일이다.
    """
    if stock < 0:
        raise ValueError(f"재고는 0 이상이어야 합니다: {stock}")
    for agent_id, amount in requests.items():
        if amount < 0:
            raise ValueError(f"요청 마릿수는 0 이상이어야 합니다: {agent_id}={amount}")

    total = sum(requests.values())
    if total <= stock:
        return dict(requests)

    # 정수 계산만 쓴다 (소수 계산의 미세한 오차로 1마리가 달라지는 일을 막는다).
    return {agent_id: amount * stock // total for agent_id, amount in requests.items()}


def settle_money(world: World, catches: dict[str, int]) -> None:
    """판매와 생활비를 정산해 어부들의 돈을 고친다. SPEC 1장 "경제", 3장 정산의 판매·생활비.

    돈 += 잡은 마릿수 × 판매가 - 하루 생활비

    catches: 어부 id -> 실제로 잡은 마릿수 (`allocate_catch`의 결과).
             목록에 없는 살아 있는 어부는 0마리로 본다 (쉬었거나 응답을 못 한 날도 생활비는 낸다).
    이미 마을을 떠난 어부는 건너뛴다 (팔 것도, 낼 생활비도 없다).
    돈은 0 아래로 내려갈 수 있다. 빚진 날 수는 `check_bankruptcy`가 센다.
    """
    economy = world.config.economy
    alive_ids = {a.id for a in world.alive_agents()}
    for agent_id, amount in catches.items():
        if agent_id not in alive_ids:
            raise ValueError(f"마을에 없는 어부의 어획입니다: {agent_id}")
        if amount < 0:
            raise ValueError(f"잡은 마릿수는 0 이상이어야 합니다: {agent_id}={amount}")

    for agent in world.alive_agents():
        caught = catches.get(agent.id, 0)
        agent.money += caught * economy.fish_price - economy.daily_cost


def check_bankruptcy(world: World) -> list[str]:
    """빚진 날을 세고, 기준 일수에 닿은 어부를 마을에서 떠나게 한다. SPEC 1장 "파산", 3장 정산의 파산 판정.

    - 돈 < 0 이면 빚진 날 +1, 0 이상이면 0으로 되돌린다 (0원은 빚이 아니다).
    - 빚진 날이 `bankrupt_after_days`(3일)에 닿으면 그날 떠난다 (alive = False).

    `settle_money` 다음에 하루 한 번 부른다. 오늘 떠난 어부의 id를 설정 파일 순서로 돌려준다.
    """
    limit = world.config.economy.bankrupt_after_days
    left: list[str] = []
    for agent in world.alive_agents():
        agent.debt_days = agent.debt_days + 1 if agent.money < 0 else 0
        if agent.debt_days >= limit:
            agent.alive = False
            left.append(agent.id)
    return left


def remove_catch(world: World, catches: dict[str, int]) -> None:
    """잡힌 물고기를 연못 재고에서 뺀다. SPEC 3장 정산의 어획 배분 직후.

    catches: 어부 id -> 실제로 잡은 마릿수 (`allocate_catch`의 결과).
    합계가 재고보다 많거나 음수가 있으면 앞 단계의 실수이므로 `ValueError`로 멈춘다.
    """
    for agent_id, amount in catches.items():
        if amount < 0:
            raise ValueError(f"잡은 마릿수는 0 이상이어야 합니다: {agent_id}={amount}")
    total = sum(catches.values())
    if total > world.pond.stock:
        raise ValueError(f"재고({world.pond.stock})보다 많이 잡을 수 없습니다: 합계 {total}")
    world.pond.stock -= total


def check_collapse(world: World) -> bool:
    """남은 물고기가 기준(5마리)보다 적으면 연못을 붕괴로 표시한다. SPEC 1장 "붕괴".

    재생 **전에** 부른다 (붕괴한 연못은 재생하지 않는다. NOTES 2026-11-14 참고).
    한 번 붕괴하면 계속 붕괴 상태다. 붕괴했으면 True를 돌려준다.
    """
    if world.pond.stock < world.config.pond.collapse_below:
        world.pond.collapsed = True
    return world.pond.collapsed


def regenerate(world: World) -> None:
    """남은 물고기를 배수(2배)만큼 늘린다. 상한은 연못 크기(100). SPEC 1장 "재생".

    붕괴한 연못은 그대로 둔다. `check_collapse` 다음에 하루 한 번 부른다.
    """
    pond = world.pond
    if pond.collapsed:
        return
    pond.stock = min(pond.capacity, pond.stock * world.config.pond.regen_multiplier)
