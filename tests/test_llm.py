"""llm.py 최소 버전 테스트. 실제 네트워크 호출 없이 가짜 클라이언트로 확인한다."""

from types import SimpleNamespace

import pytest

from pondvillage import llm


class FakeClient:
    """openai 클라이언트 흉내. 받은 요청을 기억하고 정해진 답을 돌려준다."""

    def __init__(self, text="안녕하세요!", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)):
        self.calls = []
        self._text = text
        self._usage = usage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self._usage)


def test_chat_returns_text_tokens_and_latency():
    client = FakeClient()
    result = llm.chat([{"role": "user", "content": "hi"}], model="m", client=client)

    assert result.text == "안녕하세요!"
    assert (result.prompt_tokens, result.completion_tokens) == (7, 3)
    assert result.latency_ms >= 0
    assert client.calls[0]["model"] == "m"


def test_chat_handles_missing_usage_and_empty_content():
    client = FakeClient(text=None, usage=None)
    result = llm.chat([{"role": "user", "content": "hi"}], client=client)

    assert result.text == ""
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)


def test_chat_refuses_non_local_address():
    client = FakeClient()
    with pytest.raises(llm.PaidCallNotAllowed):
        llm.chat(
            [{"role": "user", "content": "hi"}],
            base_url="https://openrouter.ai/api/v1",
            client=client,
        )
    assert client.calls == []


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("https://openrouter.ai/api/v1", False),
        ("https://localhost.example.com/v1", False),
    ],
)
def test_is_local(base_url, expected):
    assert llm.is_local(base_url) is expected


def test_ping_refused_for_paid_address_exits_with_2(capsys):
    assert llm.main(["--ping", "--base-url", "https://openrouter.ai/api/v1"]) == 2
    assert "거절" in capsys.readouterr().err
