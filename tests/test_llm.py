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


def test_chat_json_mode_asks_for_json_object():
    client = FakeClient(text='{"action": "fish", "amount": 10, "reason": "r"}')
    llm.chat([{"role": "user", "content": "hi"}], json_mode=True, client=client)

    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_without_json_mode_sends_no_response_format():
    client = FakeClient()
    llm.chat([{"role": "user", "content": "hi"}], client=client)

    assert "response_format" not in client.calls[0]


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


# ---------------------------------------------------------------- JSON 해석 (P4)


from openai import APIConnectionError, NotFoundError


@pytest.mark.parametrize("text", [
    '{"action": "fish", "amount": 10, "reason": "r"}',
    '```json\n{"action": "fish", "amount": 10, "reason": "r"}\n```',
    '좋아요. {"action": "fish", "amount": 10, "reason": "r"} 이렇게 할게요.',
])
def test_parse_json_reads_plain_boxed_and_chatty_answers(text):
    data, error = llm.parse_json(text)
    assert data == {"action": "fish", "amount": 10, "reason": "r"}
    assert error is None


@pytest.mark.parametrize("text", ["", "열 마리 잡을래요", "[1, 2]", '{"action": "fish",'])
def test_parse_json_fails_without_crashing(text):
    data, error = llm.parse_json(text)
    assert data is None
    assert error


class SeqClient:
    """정해 둔 답(글 또는 오류)을 차례로 돌려주는 가짜 클라이언트."""

    def __init__(self, answers, models=("qwen2.5:7b",)):
        self.answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._models = models
        self.models = SimpleNamespace(list=self._list)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))], usage=usage)

    def _list(self):
        if isinstance(self._models, Exception):
            raise self._models
        return [SimpleNamespace(id=m) for m in self._models]


def conn_error():
    return APIConnectionError(request=None)


def caller(client, records=None, cap=None):
    return llm.LLMCaller(
        client=client, meter=llm.CostMeter(cap),
        record=records.append if records is not None else None,
    )


def test_ask_json_uses_json_mode_and_records_the_call():
    records = []
    client = SeqClient(['{"action": "fish", "amount": 7, "reason": "r"}'])
    reply = caller(client, records).ask_json(
        [{"role": "user", "content": "q"}], model="gemma3:4b", day=3, agent_id="a2", phase="fish"
    )
    assert reply.data["amount"] == 7 and reply.calls == 1
    assert client.calls[0]["model"] == "gemma3:4b"
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    rec = records[0]
    assert (rec.day, rec.agent_id, rec.phase, rec.model) == (3, "a2", "fish", "gemma3:4b")
    assert (rec.prompt_tokens, rec.completion_tokens, rec.cost_usd, rec.parse_error) == (50, 10, 0.0, None)
    assert '"q"' in rec.prompt and "amount" in rec.response


def test_ask_json_retries_once_after_bad_answer():
    records = []
    client = SeqClient(["잘 모르겠어요", '{"action": "inspect_pond", "reason": "r"}'])
    reply = caller(client, records).ask_json([{"role": "user", "content": "q"}], model="m")
    assert reply.data == {"action": "inspect_pond", "reason": "r"}
    assert reply.calls == 2
    # 두 번째 질문에는 첫 답과 다시 답하라는 말이 붙는다
    second = client.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "잘 모르겠어요"}
    assert second[-1]["content"] == llm.RETRY_NUDGE
    assert [r.parse_error is None for r in records] == [False, True]


def test_ask_json_gives_up_after_two_bad_answers_without_crashing():
    records = []
    reply = caller(SeqClient(["음", "글쎄"]), records).ask_json([{"role": "user", "content": "q"}], model="m")
    assert reply.data is None
    assert "두 번 모두" in reply.parse_error
    assert len(records) == 2


def test_connection_drop_is_retried_once():
    client = SeqClient([conn_error(), '{"action": "fish", "amount": 1, "reason": "r"}'])
    reply = caller(client).ask_json([{"role": "user", "content": "q"}], model="m")
    assert reply.data["amount"] == 1


def test_server_down_stops_with_llm_unavailable():
    client = SeqClient([conn_error(), conn_error()])
    with pytest.raises(llm.LLMUnavailable):
        caller(client).ask_json([{"role": "user", "content": "q"}], model="m")


def test_missing_model_error_is_not_retried():
    response = SimpleNamespace(request=None, status_code=404, headers={})
    client = SeqClient([NotFoundError("model not found", response=response, body=None)])
    with pytest.raises(llm.LLMUnavailable):
        caller(client).ask_json([{"role": "user", "content": "q"}], model="m")
    assert len(client.calls) == 1


# ---------------------------------------------------------------- 비용


def test_local_calls_cost_nothing():
    assert llm.estimate_cost("http://localhost:11434/v1", 1000, 1000) == 0.0


def test_cost_meter_stops_when_over_cap():
    meter = llm.CostMeter(1.0)
    meter.add(0.6)
    meter.add(0.4)  # 상한과 같으면 아직 괜찮다
    with pytest.raises(llm.CostCapExceeded):
        meter.add(0.01)


def test_cost_meter_without_cap_never_stops():
    meter = llm.CostMeter(None)
    meter.add(1000.0)
    assert meter.total_usd == 1000.0


# ---------------------------------------------------------------- 시작 전 점검


def test_check_ollama_passes_when_all_models_exist():
    llm.check_ollama("http://localhost:11434/v1", ["qwen2.5:7b", "gemma3:4b", "qwen2.5:7b"],
                     client=SeqClient([], models=("qwen2.5:7b", "gemma3:4b", "x")))


def test_check_ollama_names_missing_model():
    with pytest.raises(llm.LLMUnavailable, match="gemma3:4b"):
        llm.check_ollama("http://localhost:11434/v1", ["qwen2.5:7b", "gemma3:4b"],
                         client=SeqClient([], models=("qwen2.5:7b",)))


def test_check_ollama_explains_server_down():
    with pytest.raises(llm.LLMUnavailable, match="WSL"):
        llm.check_ollama("http://localhost:11434/v1", ["qwen2.5:7b"], client=SeqClient([], models=conn_error()))
