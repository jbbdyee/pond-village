"""규칙 엔진: 어획 배분, 재생, 정산, 사망 판정 (LLM 없음).

여기 있는 함수는 입력만 보고 답을 낸다. LLM도, 무작위도 쓰지 않는다.
"""

from __future__ import annotations


def allocate_catch(requests: dict[str, int], stock: int) -> dict[str, int]:
    """어부들이 요청한 마릿수를 받아, 실제로 잡히는 마릿수를 돌려준다. SPEC 3장.

    - 요청 합계 ≤ 재고: 요청대로 준다.
    - 요청 합계 > 재고: 요청 비율로 나누고 소수점은 내린다.
      내림 때문에 남는 물고기는 아무에게도 주지 않는다 (연못에 그대로 남는다).

    requests: 어부 id -> 요청 마릿수. 상한(20마리 등)은 이미 `actions.py`에서
              잘라 낸 값이라고 본다. 연못을 확인만 한 어부는 0으로 넣는다.
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
