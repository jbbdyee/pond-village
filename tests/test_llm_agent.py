"""llm_agent.py 테스트: 질문지에 무엇이 들어가고 빠지는지, 어부별 모델, 해석 실패 처리. 진짜 LLM은 부르지 않는다."""

import copy
from pathlib import Path

from pondvillage.actions import NONE, PARSE_ERROR_KEY, validate_fish_action
from pondvillage.agents.base import Observation
from pondvillage.agents.llm_agent import LLMAgent, build_llm_agents, fish_system_prompt
from pondvillage.llm import JsonReply
from pondvillage.world import config_from_dict, load_config

BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
CONFIG = load_config(BASE_YAML)


class FakeCaller:
    """LLMCaller 흉내: 받은 요청을 기억하고 정해 둔 답을 돌려준다."""

    def __init__(self, data=None, parse_error=None):
        self.data = data if data is not None or parse_error else {"action": "fish", "amount": 9, "reason": "r"}
        self.parse_error = parse_error
        self.asked = []

    def ask_json(self, messages, *, model, day=None, agent_id=None, phase=None):
        self.asked.append({"messages": messages, "model": model, "day": day, "agent_id": agent_id, "phase": phase})
        return JsonReply(self.data, self.parse_error, 1)


def obs(**kw):
    base = dict(day=3, agent_id="agent_01", money=17, stock_estimate=88, yesterday_total_catch=42)
    base.update(kw)
    return Observation(**base)


def user_text(caller):
    return caller.asked[-1]["messages"][1]["content"]


def system_text(caller):
    return caller.asked[-1]["messages"][0]["content"]


def test_system_prompt_numbers_come_from_config():
    text = fish_system_prompt(CONFIG)
    for piece in ["5명", "최대 100마리", "2배", "5마리보다 적어지면", "생활비는 매일 5원", "3일", "최대 20마리", "최대 10마리"]:
        assert piece in text, piece
    assert "{" in text and "inspect_pond" in text  # JSON 모양 안내는 남는다


def test_system_prompt_hides_sustainable_amount_and_total_days():
    text = fish_system_prompt(CONFIG)
    assert "50" not in text  # 지속 가능한 하루 총 어획 (스스로 알아내야 함)
    assert "30" not in text  # 전체 일수


def test_name_only_when_persona_disabled():
    caller = FakeCaller()
    agents = build_llm_agents(CONFIG, caller, {s.id: "qwen2.5:7b" for s in CONFIG.agents})
    agents["agent_01"].decide_fish(obs())
    text = user_text(caller)
    assert "만석" in text
    assert "식구" not in text  # 성격 문구가 들어가지 않는다


def test_persona_added_when_enabled():
    raw = copy.deepcopy(_raw_base())
    raw["llm"]["persona_enabled"] = True
    config = config_from_dict(raw)
    caller = FakeCaller()
    agents = build_llm_agents(config, caller, {s.id: "m" for s in config.agents})
    agents["agent_01"].decide_fish(obs())
    assert "식구가 여섯" in user_text(caller)


def test_morning_facts_are_in_the_prompt_but_not_exact_stock():
    caller = FakeCaller()
    LLMAgent("agent_01", "만석", "m", CONFIG, caller).decide_fish(obs())
    text = user_text(caller)
    for piece in ["3일째", "약 88마리", "어제 마을 전체 어획 42마리", "17원"]:
        assert piece in text, piece
    assert "정확한 재고" not in text
    assert caller.asked[-1]["phase"] == "fish"


def test_first_day_says_no_yesterday():
    caller = FakeCaller()
    LLMAgent("agent_01", "만석", "m", CONFIG, caller).decide_fish(obs(day=1, yesterday_total_catch=None))
    assert "첫날" in user_text(caller)


def test_after_inspect_prompt_shows_exact_stock_and_cap():
    caller = FakeCaller()
    LLMAgent("agent_01", "만석", "m", CONFIG, caller).decide_fish(obs(exact_stock=61))
    text = user_text(caller)
    assert "정확한 재고는 61마리" in text and "최대 10마리" in text
    assert caller.asked[-1]["phase"] == "fish_after_inspect"


def test_each_fisher_asks_with_own_model():
    caller = FakeCaller()
    models = {s.id: ("gemma3:4b" if i % 2 else "qwen2.5:7b") for i, s in enumerate(CONFIG.agents)}
    agents = build_llm_agents(CONFIG, caller, models)
    for agent_id, agent in agents.items():
        agent.decide_fish(obs(agent_id=agent_id))
    assert [a["model"] for a in caller.asked] == list(models.values())
    assert [a["agent_id"] for a in caller.asked] == list(models)


def test_prompt_does_not_reveal_models():
    caller = FakeCaller()
    LLMAgent("agent_01", "만석", "gemma3:4b", CONFIG, caller).decide_fish(obs())
    assert "gemma" not in system_text(caller) + user_text(caller)


def test_answer_is_passed_through_for_validation():
    caller = FakeCaller(data={"action": "fish", "amount": 30, "reason": "많이"})
    raw = LLMAgent("agent_01", "만석", "m", CONFIG, caller).decide_fish(obs())
    decision = validate_fish_action(raw, 20)
    assert (decision.amount, decision.clamped) == (20, True)


def test_parse_failure_becomes_do_nothing_with_reason():
    caller = FakeCaller(parse_error="두 번 모두 해석 실패: JSON 아님")
    raw = LLMAgent("agent_01", "만석", "m", CONFIG, caller).decide_fish(obs())
    assert raw == {PARSE_ERROR_KEY: "두 번 모두 해석 실패: JSON 아님"}
    decision = validate_fish_action(raw, 20)
    assert (decision.kind, decision.amount) == (NONE, 0)
    assert decision.parse_error == "두 번 모두 해석 실패: JSON 아님"


def _raw_base():
    import yaml

    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
