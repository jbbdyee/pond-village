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

# LLM 답을 JSON으로 읽지 못했을 때 어부가 돌려주는 dict의 열쇠. 값은 실패 이유.
# 이 열쇠가 있으면 모양 검사 없이 "아무것도 안 함" + 그 이유를 parse_error로 남긴다.
PARSE_ERROR_KEY = "_parse_error"


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
    if raw.get(PARSE_ERROR_KEY):
        return FishDecision(NONE, 0, "", parse_error=str(raw[PARSE_ERROR_KEY]))

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


# ---------------------------------------------------------------- 광장 (say, P6)

SAY = "say"
TO_ALL = "all"  # 모두에게. 그 밖의 to는 받는 어부 id (귓속말)
_ALL_WORDS = {"all", "모두", "모두에게", "전체", "everyone"}


@dataclass(frozen=True)
class SayDecision:
    """검사를 마친 광장 한 마디. silent면 말하지 않은 것 (해석 실패)."""

    to: str  # TO_ALL 또는 받는 어부 id. silent면 TO_ALL
    message: str  # 잘라낸 뒤의 말. silent면 빈 글
    reason: str
    clamped: bool = False  # 글이 길어 잘렸거나, 받는 사람을 알 수 없어 "모두에게"로 바꿨는지
    parse_error: str | None = None

    @property
    def silent(self) -> bool:
        return self.parse_error is not None

    @property
    def whisper(self) -> bool:
        return self.to != TO_ALL


def validate_say_action(
    raw: object,
    speaker_id: str,
    name_to_id: dict[str, str],
    listeners: set[str],
    max_chars: int,
) -> SayDecision:
    """어부의 광장 답을 검사한다 (SPEC 4장 say).

    - message가 비었거나 글이 아니면 "말 안 함" + parse_error. action이 있는데 say가 아니어도 실패.
    - to는 "all"(또는 "모두") / 이웃 이름 / 이웃 id를 받는다. LLM에게는 이름만 알려 주므로 이름이 보통이다.
      자기 자신·모르는 이름·떠난 사람이면 "모두에게"로 바꾸고 clamped로 표시한다 (말은 살린다).
    - message는 max_chars에서 자르고 clamped로 표시한다.
    listeners: 지금 광장에 있는(남아 있는) 어부 id들.
    """
    def fail(why: str, reason: str = "") -> SayDecision:
        return SayDecision(TO_ALL, "", reason, parse_error=why)

    if not isinstance(raw, dict):
        return fail(f"dict가 아님: {type(raw).__name__}")
    if raw.get(PARSE_ERROR_KEY):
        return fail(str(raw[PARSE_ERROR_KEY]))

    reason = raw.get("reason", "")
    reason = reason if isinstance(reason, str) else str(reason)
    action = raw.get("action", SAY)
    if action != SAY:
        return fail(f"모르는 행동: {action!r}", reason)
    message = raw.get("message")
    if not isinstance(message, str) or not message.strip():
        return fail(f"message가 글이 아님: {message!r}", reason)
    message = message.strip()
    clamped = len(message) > max_chars
    message = message[:max_chars]

    to_raw = raw.get("to", TO_ALL)
    key = str(to_raw).strip() if to_raw is not None else TO_ALL
    if key.lower() in _ALL_WORDS or key == "":
        to = TO_ALL
    else:
        target = name_to_id.get(key, key if key in listeners else None)
        if target is None or target == speaker_id or target not in listeners:
            to, clamped = TO_ALL, True
        else:
            to = target
    return SayDecision(to, message, reason, clamped=clamped)
