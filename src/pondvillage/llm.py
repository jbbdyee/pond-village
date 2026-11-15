"""LLM 호출은 전부 이 모듈에서만 한다 (지침 규칙 2).

- `chat`: 한 번 묻고 글로 받는다. 로컬 주소가 아니면 거절한다 (유료 호출은 아직 닫혀 있다).
- `parse_json`: 받은 글을 dict로 바꾼다. ```json 상자나 앞뒤 군말이 있어도 첫 { ~ 마지막 }을 읽는다.
- `LLMCaller.ask_json`: 어부가 쓰는 입구. JSON 모드로 묻고, 해석에 실패하면 1번 더 묻고,
  그래도 실패하면 data=None과 parse_error를 돌려준다 (절대 멈추지 않는다, 지침 규칙 6).
  모든 호출을 `record`로 넘겨 `llm_calls`에 남기고, 비용 합계가 상한을 넘으면 멈춘다.
- `check_ollama`: 러너가 시작할 때 서버가 살아 있고 모델이 다 있는지 5초 안에 확인한다.
  (WSL이 쉬면 Ollama가 꺼져 호출마다 120초씩 멈추기 때문)

실행: uv run python -m pondvillage.llm --ping
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# 개발용 기본 모델 (2026-11-15 결정, configs/base.yaml과 같게 둔다).
DEFAULT_MODEL = "qwen2.5:7b"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
TIMEOUT_SECONDS = 120.0
CHECK_TIMEOUT_SECONDS = 5.0

# 해석에 실패했을 때 두 번째로 물을 때 덧붙이는 말
RETRY_NUDGE = "방금 답은 JSON으로 읽을 수 없었다. 다른 글 없이 JSON 객체 하나로만 다시 답하라."


class PaidCallNotAllowed(RuntimeError):
    """로컬이 아닌 주소로 호출하려 할 때. 유료 호출은 아직 열지 않았다."""


class LLMUnavailable(RuntimeError):
    """서버에 닿지 않거나 모델이 없을 때. 실행을 깔끔히 멈춘다 (크래시 대신 status=error)."""


class CostCapExceeded(RuntimeError):
    """비용 합계가 상한을 넘었을 때. 러너가 status=aborted_cost로 멈춘다 (SPEC 8장)."""


@dataclass(frozen=True)
class LLMResult:
    """호출 한 번의 결과."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


@dataclass(frozen=True)
class CallRecord:
    """`llm_calls` 한 줄. 무엇을 묻고 무엇을 받았는지까지 남긴다."""

    day: int | None
    agent_id: str | None
    phase: str | None
    model: str
    prompt: str  # 보낸 messages를 JSON 글로
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    parse_error: str | None  # 이 답을 JSON으로 못 읽었으면 그 이유


@dataclass(frozen=True)
class JsonReply:
    """`ask_json`의 결과. data가 None이면 두 번 다 해석 실패."""

    data: dict | None
    parse_error: str | None
    calls: int  # 실제로 부른 횟수 (1 또는 2)


def is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in LOCAL_HOSTS


def make_client(base_url: str, timeout: float = TIMEOUT_SECONDS) -> OpenAI:
    # Ollama는 키를 검사하지 않지만 openai 패키지는 빈 키를 거절해서 아무 값이나 넣는다.
    # 재시도는 우리가 직접 하므로 패키지 자체 재시도는 끈다 (꺼진 서버에서 오래 기다리지 않게).
    return OpenAI(base_url=base_url, api_key="ollama", timeout=timeout, max_retries=0)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    json_mode: bool = False,
    client: OpenAI | None = None,
) -> LLMResult:
    """LLM에 한 번 묻는다. 로컬 주소가 아니면 호출하지 않고 멈춘다.

    json_mode: True면 "JSON 객체 하나로만 답하라"고 서버에 요청한다
               (OpenAI 호환 `response_format={"type": "json_object"}`, Ollama도 지원).
    """
    if not is_local(base_url):
        raise PaidCallNotAllowed(
            f"로컬 주소가 아닙니다: {base_url} (유료 호출은 --paid와 cost_cap_usd가 생긴 뒤에만)"
        )
    client = client or make_client(base_url)
    started = time.perf_counter()
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **extra
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    usage = response.usage
    return LLMResult(
        text=response.choices[0].message.content or "",
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------- JSON 해석


def parse_json(text: str) -> tuple[dict | None, str | None]:
    """글 → (dict, None) 또는 (None, 실패 이유).

    그대로 읽어 보고, 안 되면 첫 { ~ 마지막 }만 잘라 다시 읽는다 (```json 상자, 앞뒤 군말 대비).
    JSON이어도 객체(dict)가 아니면 실패로 본다.
    """
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group(0))
    error = "빈 답" if not text.strip() else "JSON 아님"
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            error = f"JSON 아님: {e.msg}"
            continue
        if isinstance(data, dict):
            return data, None
        error = f"JSON 객체가 아님: {type(data).__name__}"
    return None, error


# ---------------------------------------------------------------- 비용


def is_free(base_url: str) -> bool:
    """로컬 Ollama는 0원. 유료 모델 가격표는 본 실험 모델을 고른 뒤(P9 전) 붙인다."""
    return is_local(base_url)


def estimate_cost(base_url: str, prompt_tokens: int, completion_tokens: int) -> float:
    if is_free(base_url):
        return 0.0
    raise PaidCallNotAllowed("유료 모델 가격표가 아직 없습니다")


class CostMeter:
    """실행 1회의 비용 합계. 상한을 넘는 순간 CostCapExceeded."""

    def __init__(self, cap_usd: float | None) -> None:
        self.cap_usd = cap_usd
        self.total_usd = 0.0

    def add(self, cost_usd: float) -> None:
        self.total_usd += cost_usd
        if self.cap_usd is not None and self.total_usd > self.cap_usd:
            raise CostCapExceeded(f"비용 {self.total_usd:.4f} USD가 상한 {self.cap_usd} USD를 넘었습니다")


# ---------------------------------------------------------------- 어부가 쓰는 입구


class LLMCaller:
    """어부들이 함께 쓰는 LLM 창구. 주소·온도·비용 계량기·기록원을 한곳에 묶는다.

    어부마다 다른 것은 `model` 하나뿐이라 `ask_json`에 넘긴다.
    record: CallRecord를 받아 저장하는 함수 (러너가 `RunLogger.log_llm_call`을 꽂는다). 없으면 기록 안 함.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        *,
        meter: CostMeter | None = None,
        record: Callable[[CallRecord], None] | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.base_url = base_url
        self.temperature = temperature
        self.meter = meter or CostMeter(None)
        self.record = record
        self.client = client

    def _call(self, messages: list[dict[str, str]], model: str) -> LLMResult:
        """한 번 부른다. 연결이 끊기면 1번 더, 그래도 안 되면 LLMUnavailable."""
        last: Exception | None = None
        for _ in range(2):
            try:
                return chat(
                    messages, model=model, base_url=self.base_url, temperature=self.temperature,
                    json_mode=True, client=self.client,
                )
            except (APIConnectionError, APITimeoutError) as e:
                last = e
            except OpenAIError as e:  # 모델 없음 등: 다시 불러도 같다
                raise LLMUnavailable(f"{model} 호출 실패: {e}") from e
        raise LLMUnavailable(
            f"{self.base_url} 의 {model} 에 닿지 않습니다: {last}. Ollama(WSL)가 켜져 있는지 확인하세요."
        ) from last

    def ask_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        day: int | None = None,
        agent_id: str | None = None,
        phase: str | None = None,
    ) -> JsonReply:
        """JSON 하나를 받아 온다. 해석 실패면 1번 더 묻고, 그래도 실패면 data=None (멈추지 않음)."""
        convo = list(messages)
        error: str | None = None
        for attempt in (1, 2):
            result = self._call(convo, model)
            data, error = parse_json(result.text)
            cost = estimate_cost(self.base_url, result.prompt_tokens, result.completion_tokens)
            if self.record is not None:
                self.record(CallRecord(
                    day=day, agent_id=agent_id, phase=phase, model=model,
                    prompt=json.dumps(convo, ensure_ascii=False), response=result.text,
                    prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                    latency_ms=result.latency_ms, cost_usd=cost, parse_error=error,
                ))
            self.meter.add(cost)  # 기록한 뒤에 검사: 상한을 넘긴 호출도 남는다
            if data is not None:
                return JsonReply(data, None, attempt)
            convo = convo + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": RETRY_NUDGE},
            ]
        return JsonReply(None, f"두 번 모두 해석 실패: {error}", 2)


# ---------------------------------------------------------------- 시작 전 점검


def check_ollama(base_url: str, models: list[str], client: OpenAI | None = None) -> None:
    """서버가 살아 있고 쓸 모델이 다 있는지 확인한다. 문제가 있으면 LLMUnavailable.

    모델이 없어도 받지 않는다 (다운로드는 사용자에게 먼저 묻는다).
    """
    client = client or make_client(base_url, timeout=CHECK_TIMEOUT_SECONDS)
    try:
        available = {m.id for m in client.models.list()}
    except OpenAIError as e:
        raise LLMUnavailable(
            f"{base_url} 에 닿지 않습니다 ({type(e).__name__}).\n"
            "WSL(Ubuntu)이 쉬면 Ollama가 꺼집니다. Ubuntu 창을 열어 Ollama를 켠 뒤 다시 실행하세요."
        ) from e
    missing = [m for m in dict.fromkeys(models) if m not in available]
    if missing:
        raise LLMUnavailable(
            f"Ollama에 없는 모델: {', '.join(missing)} (있는 모델: {', '.join(sorted(available)) or '없음'})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="연못마을 LLM 연결 확인")
    parser.add_argument("--ping", action="store_true", help="모델에 인사를 보내고 답을 출력")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if not args.ping:
        parser.print_help()
        return 0

    try:
        result = chat(
            [{"role": "user", "content": "한 문장으로 짧게 인사해 줘."}],
            model=args.model,
            base_url=args.base_url,
        )
    except PaidCallNotAllowed as error:
        print(f"[거절] {error}", file=sys.stderr)
        return 2
    except OpenAIError as error:
        print(
            f"[실패] {args.base_url} 의 {args.model} 호출에 실패했습니다: {error}\n"
            "Ollama가 켜져 있는지(Ubuntu가 쉬고 있지 않은지), 모델 이름이 맞는지 확인하세요.",
            file=sys.stderr,
        )
        return 1

    print(f"모델: {result.model} ({args.base_url})")
    print(f"응답: {result.text.strip()}")
    print(
        f"토큰: 입력 {result.prompt_tokens} / 출력 {result.completion_tokens}"
        f" · 걸린 시간: {result.latency_ms}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
