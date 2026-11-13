"""LLM 호출은 전부 이 모듈에서만 한다.

최소 버전: OpenAI 호환 주소(기본은 로컬 Ollama)로 한 번 묻고 답을 받는다.
호출 기록(`llm_calls`)·JSON 파싱·재시도·비용 상한은 P3~P4에서 붙인다.

실행: uv run python -m pondvillage.llm --ping
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from openai import OpenAI, OpenAIError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
# 임시 기본값. 개발용 모델이 정해지면 configs/base.yaml에서 읽는다.
DEFAULT_MODEL = "qwen2.5vl:7b"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
TIMEOUT_SECONDS = 120.0


class PaidCallNotAllowed(RuntimeError):
    """로컬이 아닌 주소로 호출하려 할 때. 유료 호출은 아직 열지 않았다."""


@dataclass(frozen=True)
class LLMResult:
    """호출 한 번의 결과. 나중에 `llm_calls` 테이블에 그대로 기록할 값들."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in LOCAL_HOSTS


def make_client(base_url: str) -> OpenAI:
    # Ollama는 키를 검사하지 않지만 openai 패키지는 빈 키를 거절해서 아무 값이나 넣는다.
    return OpenAI(base_url=base_url, api_key="ollama", timeout=TIMEOUT_SECONDS)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    client: OpenAI | None = None,
) -> LLMResult:
    """LLM에 한 번 묻는다. 로컬 주소가 아니면 호출하지 않고 멈춘다."""
    if not is_local(base_url):
        raise PaidCallNotAllowed(
            f"로컬 주소가 아닙니다: {base_url} (유료 호출은 --paid와 cost_cap_usd가 생긴 뒤에만)"
        )
    client = client or make_client(base_url)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
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
