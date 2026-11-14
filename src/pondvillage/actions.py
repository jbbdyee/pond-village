"""행동 정의(도구 스키마)와 검증.

어부(봇이든 LLM이든)가 돌려준 dict를 받아, 러너와 규칙 엔진이 믿고 쓸 수 있는 값으로 바꾼다.
- 범위를 벗어난 마릿수는 잘라내고 `clamped`로 표시한다 (SPEC 4장).
- 모양이 틀린 답은 "아무것도 안 함"으로 바꾸고 `parse_error`로 표시한다. 절대 멈추지 않는다 (지침 규칙 6).

여기서는 "몇 마리를 요청한 것으로 볼지"만 정한다. 실제로 몇 마리를 받는지는 `rules.allocate_catch`가 정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pondvillage.world import EconomyConfig

FISH = "fish"
INSPECT = "inspect_pond"
NONE = "none"  # 아무것도 안 함 (해석 실패, 또는 확인을 두 번 고른 경우)


@dataclass(frozen=True)
class FishDecision:
    """검사를 마친 어획 페이즈 행동 하나."""

    kind: str  # FISH / INSPECT / NONE
    amount: int  # 요청 마릿수 (잘라낸 뒤). fish가 아니면 0
    reason: str
    clamped: bool = False  # 범위를 벗어나 잘렸는지
    parse_error: str | None = None  # 모양이 틀렸으면 무엇이 틀렸는지


def catch_cap(economy: EconomyConfig, inspected: bool) -> int:
    """오늘 이 어부가 요청할 수 있는 최대 마릿수. 기본 20, 연못을 확인한 날은 절반(10). SPEC 1장.

    소수가 나오면 내린다 (상한을 넘겨 주지 않기 위해).
    """
    if inspected:
        return math.floor(economy.max_catch_per_day * economy.inspect_catch_factor)
    return economy.max_catch_per_day


def _as_int(value: object) -> int | None:
    """정수로 볼 수 있으면 정수로, 아니면 None. 10.0 같은 값은 받아 주고, True/False·"10"은 받지 않는다."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def validate_fish_action(raw: object, cap: int, allow_inspect: bool = True) -> FishDecision:
    """어부의 어획 답을 검사한다.

    raw:           어부가 돌려준 dict (SPEC 4장 `fish` 또는 `inspect_pond` 모양이어야 한다)
    cap:           오늘 요청할 수 있는 최대 마릿수 (`catch_cap`)
    allow_inspect: False면 연못 확인을 받지 않는다. 확인한 뒤 다시 물었을 때 쓴다.
                   이때 또 확인을 고르면 해석 실패가 아니라 "아무것도 안 함"(0마리)으로 본다.
    """
    if not isinstance(raw, dict):
        return FishDecision(NONE, 0, "", parse_error=f"dict가 아님: {type(raw).__name__}")

    reason = raw.get("reason", "")
    reason = reason if isinstance(reason, str) else str(reason)
    action = raw.get("action")

    if action == INSPECT:
        if allow_inspect:
            return FishDecision(INSPECT, 0, reason)
        return FishDecision(NONE, 0, reason)

    if action != FISH:
        return FishDecision(NONE, 0, reason, parse_error=f"모르는 행동: {action!r}")

    amount = _as_int(raw.get("amount"))
    if amount is None:
        return FishDecision(NONE, 0, reason, parse_error=f"마릿수가 정수가 아님: {raw.get('amount')!r}")

    clipped = min(max(amount, 0), cap)
    return FishDecision(FISH, clipped, reason, clamped=clipped != amount)
