"""P6 광장 대화 테스트: 말 검사, 누가 무엇을 듣는지, 순서 섞기, 켜기/끄기, 기록, 질문지."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from pondvillage import llm, memory
from pondvillage.actions import PARSE_ERROR_KEY, TO_ALL, validate_say_action
from pondvillage.agents.base import BaseAgent, Observation, SayContext
from pondvillage.agents.llm_agent import build_llm_agents, fish_user_prompt, say_user_prompt
from pondvillage.agents.scripted import CooperativeBot
from pondvillage.logger import RunLogger, open_db
from pondvillage.memory import PlazaMessage
from pondvillage.runner import END_DAYS, format_plaza, make_rng, run, run_day
from pondvillage.world import Beliefs, World, config_from_dict, load_config

BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

RAW = {
    "pond": {"capacity": 100, "initial_stock": 100, "regen_multiplier": 2, "collapse_below": 5,
             "estimate_noise": [0.8, 1.2]},
    "economy": {"start_money": 20, "fish_price": 1, "daily_cost": 5, "max_catch_per_day": 20,
                "inspect_catch_factor": 0.5, "bankrupt_after_days": 3},
    "run": {"days": 30, "seed": 1},
    "talk": {"enabled": True, "max_chars": 200},
    "agents": [{"id": f"a{i}", "name": f"어부{i}", "persona": "어부"} for i in range(1, 6)],
}
IDS = [f"a{i}" for i in range(1, 6)]
NAMES = {f"a{i}": f"어부{i}" for i in range(1, 6)}
NAME_TO_ID = {v: k for k, v in NAMES.items()}
LISTENERS = set(IDS)


def make_config(talk=True, llm_on=False, **changes):
    raw = copy.deepcopy(RAW)
    raw["talk"]["enabled"] = talk
    for section, values in changes.items():
        raw[section].update(values)
    if llm_on:
        raw["llm"] = {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "cost_cap_usd": 1.0}
    return config_from_dict(raw)


def say(raw, speaker="a1", max_chars=200):
    return validate_say_action(raw, speaker, NAME_TO_ID, LISTENERS, max_chars)


# ---------------------------------------------------------------- 말 검사 (actions.py)


def test_say_to_all_is_accepted():
    for to in ["all", "모두", "ALL", "", None]:
        d = say({"action": "say", "to": to, "message": " 같이 아껴 잡자 ", "reason": "r"})
        assert (d.to, d.message, d.clamped, d.silent) == (TO_ALL, "같이 아껴 잡자", False, False), to


def test_missing_to_and_missing_action_mean_say_to_all():
    d = say({"message": "안녕"})
    assert d.to == TO_ALL and not d.silent


def test_whisper_by_name_or_id_becomes_id():
    assert say({"action": "say", "to": "어부3", "message": "비밀"}).to == "a3"
    d = say({"action": "say", "to": "a4", "message": "비밀"})
    assert d.to == "a4" and d.whisper and not d.clamped


def test_unknown_self_or_departed_recipient_falls_back_to_all():
    for to in ["철수", "어부1", "a1"]:
        d = say({"action": "say", "to": to, "message": "음"})
        assert (d.to, d.clamped, d.silent) == (TO_ALL, True, False), to
    d = validate_say_action({"to": "어부5", "message": "음"}, "a1", NAME_TO_ID, {"a1", "a2"}, 200)
    assert (d.to, d.clamped) == (TO_ALL, True)  # 어부5는 이미 떠났다


def test_long_message_is_cut_and_marked():
    d = say({"action": "say", "to": "all", "message": "가" * 250}, max_chars=200)
    assert len(d.message) == 200 and d.clamped


def test_bad_say_answers_mean_silent_with_parse_error():
    for raw in ["말", {"action": "say", "message": ""}, {"action": "say", "message": 3},
                {"action": "fish", "amount": 10}, {PARSE_ERROR_KEY: "두 번 모두 실패"}]:
        d = say(raw)
        assert d.silent and d.parse_error and d.message == "", raw


# ---------------------------------------------------------------- 누가 듣나 (memory.py)

MSGS = [
    PlazaMessage("a2", TO_ALL, "아껴 잡자"),
    PlazaMessage("a3", "a1", "너만 알아"),
    PlazaMessage("a1", "a4", "나도 비밀"),
]


def test_whisper_is_heard_only_by_sender_and_receiver():
    assert [m.heard_by("a5") for m in MSGS] == [True, False, False]
    assert [m.heard_by("a1") for m in MSGS] == [True, True, True]
    assert [m.heard_by("a4") for m in MSGS] == [True, False, True]


def test_heard_lines_name_who_and_to_whom():
    assert memory.heard_lines(MSGS, "a1", NAMES) == [
        '어부2(모두에게): "아껴 잡자"',
        '어부3(나에게 귓속말): "너만 알아"',
        '나(어부4에게 귓속말): "나도 비밀"',
    ]
    assert memory.heard_text(MSGS, "a5", NAMES) == '어부2(모두에게): "아껴 잡자"'


def test_describe_plaza_is_one_line_or_nothing():
    assert memory.describe_plaza(3, MSGS, "a5", NAMES) == '3일째 광장: 어부2(모두에게): "아껴 잡자"'
    assert memory.describe_plaza(3, [], "a5", NAMES) is None


def test_trust_line_only_when_asked_and_only_for_remaining_neighbors():
    world = World.from_config(make_config())
    a = world.get_agent("a1")
    a.trust = {"a2": 0.8, "a3": 0.25}
    assert "이웃 신뢰" not in memory.memory_text(a, 1)  # 대화를 끈 마을은 넣지 않는다
    text = memory.memory_text(a, 1, trust_names={"a2": "어부2"})  # a3은 떠났다
    assert "[지난 일기에 적은 이웃 신뢰 (0~1)]\n어부2 0.8" in text and "어부3" not in text
    a.trust = {}
    assert memory.memory_text(a, 1, trust_names={"a2": "어부2"}) == ""


# ---------------------------------------------------------------- 설정 (world.py)


def test_talk_is_off_when_config_has_no_talk_section():
    raw = copy.deepcopy(RAW)
    del raw["talk"]
    assert config_from_dict(raw).talk.enabled is False


def test_base_yaml_turns_talk_on_with_200_chars():
    talk = load_config(BASE_YAML).talk
    assert (talk.enabled, talk.max_chars) == (True, 200)


# ---------------------------------------------------------------- 광장 진행 (runner.py)


class Talker(BaseAgent):
    """정해 둔 말을 하고, 말할 때·고기 잡을 때 본 것을 기록하는 가짜 어부."""

    def __init__(self, agent_id, reply=None):
        super().__init__(agent_id)
        self.reply = reply or (lambda ctx: {"action": "say", "to": "all", "message": f"{agent_id} 인사"})
        self.say_seen: list[SayContext] = []
        self.fish_seen: list[Observation] = []

    def say(self, ctx):
        self.say_seen.append(ctx)
        return self.reply(ctx)

    def decide_fish(self, obs):
        self.fish_seen.append(obs)
        return {"action": "fish", "amount": 10, "reason": ""}


def talkers(**replies):
    return {i: Talker(i, replies.get(i)) for i in IDS}


def test_talk_off_skips_plaza_entirely():
    agents = talkers()
    world = World.from_config(make_config(talk=False))
    result = run(world, agents, seed=1, days=3)
    assert all(d.says == [] for d in result.days)
    assert all(a.say_seen == [] for a in agents.values())
    assert all(o.heard == "" for a in agents.values() for o in a.fish_seen)


def test_speaking_order_is_seeded_and_reshuffled_each_day():
    def orders(seed):
        world = World.from_config(make_config())
        return [[s.agent_id for s in d.says] for d in run(world, talkers(), seed=seed, days=10).days]

    first = orders(1)
    assert first == orders(1)  # 같은 시드면 같은 순서
    assert all(sorted(o) == IDS for o in first)  # 모두 한 번씩
    assert len({tuple(o) for o in first}) > 1  # 날마다 다시 섞는다
    assert first != orders(2)


def test_later_speaker_hears_public_words_and_own_whispers_only():
    world = World.from_config(make_config())
    order = list(world.alive_agents())
    make_rng(1, "talk_order").shuffle(order)
    first, second, third = (a.id for a in order[:3])
    replies = {
        first: lambda ctx: {"to": "all", "message": "첫 마디"},
        second: lambda ctx: {"to": NAMES[third], "message": "세 번째에게만"},
    }
    agents = talkers(**replies)
    others = [i for i in IDS if i not in (first, second, third)]  # 귓속말과 상관없는 두 사람
    result = run_day(world, agents, make_rng(1, "estimate"), talk_rng=make_rng(1, "talk_order"))
    assert [s.agent_id for s in result.says] == [a.id for a in order]
    assert agents[first].say_seen[0].heard == ""
    assert agents[second].say_seen[0].heard == '어부' + first[1] + '(모두에게): "첫 마디"'
    assert '나에게 귓속말): "세 번째에게만"' in agents[third].say_seen[0].heard
    for other in others:
        assert "세 번째에게만" not in agents[other].say_seen[0].heard
        assert "첫 마디" in agents[other].fish_seen[0].heard
        assert "세 번째에게만" not in agents[other].fish_seen[0].heard


def test_fishers_hear_todays_plaza_and_remember_it_next_day():
    whisper = {"a2": lambda ctx: {"to": "어부1", "message": "10마리만 잡자"}}
    agents = talkers(**whisper)
    world = World.from_config(make_config())
    run(world, agents, seed=1, days=2)
    assert '어부2(나에게 귓속말): "10마리만 잡자"' in agents["a1"].fish_seen[0].heard
    assert "10마리만" not in agents["a3"].fish_seen[0].heard
    day2_a1 = agents["a1"].fish_seen[1].memory
    assert '1일째 광장: ' in day2_a1 and "10마리만 잡자" in day2_a1
    assert "10마리만" not in agents["a3"].fish_seen[1].memory
    assert "10마리만 잡자" in agents["a2"].fish_seen[1].memory  # 보낸 사람도 기억한다


def test_bots_stay_silent_even_with_talk_on():
    world = World.from_config(make_config())
    result = run(world, {i: CooperativeBot(i) for i in IDS}, seed=1, days=3)
    assert result.end_reason == END_DAYS and all(d.says == [] for d in result.days)


def test_bad_say_answer_is_silent_but_day_goes_on():
    agents = talkers(a1=lambda ctx: {PARSE_ERROR_KEY: "두 번 모두 실패"})
    world = World.from_config(make_config())
    d = run_day(world, agents, make_rng(1, "estimate"))
    a1 = next(s for s in d.says if s.agent_id == "a1")
    assert a1.decision.silent and len(d.says) == 5 and len(d.messages) == 4
    assert d.total_catch == 50
    assert "답을 읽지 못해" in format_plaza(world, d)


def test_departed_fisher_is_not_asked_to_talk():
    world = World.from_config(make_config())
    world.get_agent("a5").alive = False
    agents = talkers()
    d = run_day(world, agents, make_rng(1, "estimate"))
    assert agents["a5"].say_seen == [] and len(d.says) == 4
    assert all("어부5" not in a.say_seen[0].neighbors for i, a in agents.items() if i != "a5")


def test_trust_reaches_prompts_only_when_talk_is_on():
    for talk in (True, False):
        agents = talkers()
        world = World.from_config(make_config(talk=talk))
        world.get_agent("a1").trust = {"a2": 0.9}
        run_day(world, agents, make_rng(1, "estimate"))
        assert ("어부2 0.9" in agents["a1"].fish_seen[0].memory) is talk


# ---------------------------------------------------------------- 기록 (logger.py)


def test_messages_and_say_actions_are_logged(tmp_path):
    agents = talkers(
        a2=lambda ctx: {"to": "어부1", "message": "비밀"},
        a3=lambda ctx: "이상한 답",
    )
    world = World.from_config(make_config())
    conn = open_db(tmp_path / "t.db")
    logger = RunLogger.start(conn, "r", config_name="t", condition="llm", seed=1, model="m")
    run(world, agents, seed=1, days=1, logger=logger)
    rows = conn.execute('SELECT "from", "to", text, is_rumor FROM messages ORDER BY rowid').fetchall()
    assert len(rows) == 4 and ("a2", "a1", "비밀", 0) in rows and all(r[0] != "a3" for r in rows)
    says = conn.execute("SELECT agent_id, action_json, parse_error FROM actions WHERE phase = 'say'").fetchall()
    assert len(says) == 5
    bad = next(r for r in says if r[0] == "a3")
    assert bad[1] == "{}" and bad[2]
    good = json.loads(next(r for r in says if r[0] == "a2")[1])
    assert good == {"action": "say", "to": "a1", "message": "비밀"}
    conn.close()


# ---------------------------------------------------------------- 질문지 (llm_agent.py)


def _say_ctx(heard=""):
    return SayContext(day=1, agent_id="a1", money=20, stock_estimate=98, yesterday_total_catch=None,
                      memory="", heard=heard, neighbors=("어부2", "어부3"))


def test_say_prompt_shows_neighbors_and_what_was_heard():
    first = say_user_prompt("어부1", None, _say_ctx())
    assert "이웃: 어부2, 어부3." in first and "네가 처음으로 말한다" in first and "98마리" in first
    later = say_user_prompt("어부1", None, _say_ctx('어부2(모두에게): "안녕"'))
    assert '[지금까지 광장에서 들은 말]\n어부2(모두에게): "안녕"' in later


def test_say_system_prompt_is_neutral_and_has_limit():
    config = make_config(llm_on=True)
    agent = build_llm_agents(config, llm.LLMCaller(client=SimpleNamespace(), meter=llm.CostMeter(1.0)),
                             {i: "qwen2.5:7b" for i in IDS})["a1"]
    system = agent.say_system
    assert "200자 이내" in system and "귓속말" in system
    plaza_part = system.split("광장에 모였다")[1]  # 규칙 설명 뒤, 광장 안내 부분
    for nudge in ["협력", "아껴", "나눠 쓰", "마리", "약속", "정하"]:
        assert nudge not in plaza_part, nudge


def test_fish_prompt_without_talk_is_unchanged():
    obs = Observation(day=1, agent_id="a1", money=20, stock_estimate=98)
    text = fish_user_prompt("어부1", None, obs, 10)
    assert "광장" not in text
    assert text.endswith("내 돈: 20원.\n오늘 무엇을 할지 JSON으로 답하라.\n")
    heard = fish_user_prompt("어부1", None, Observation(1, "a1", 20, 98, heard='어부2(모두에게): "안녕"'), 10)
    assert '[오늘 아침 광장에서 들은 말]\n어부2(모두에게): "안녕"\n오늘 무엇을' in heard


# ---------------------------------------------------------------- LLM 마을 (가짜 서버)

SAY_ALL = '{"action": "say", "to": "all", "message": "오늘은 조금만 잡자", "reason": "연못"}'
FISH10 = '{"action": "fish", "amount": 10, "reason": "나눠 쓰자"}'
DIARY = ('{"summary": "' + "광장에서 이야기했다. " * 30 + '", '
         '"beliefs": {"estimated_stock": 90, "confidence": 0.7, "sustainable_guess": 10}, '
         '"trust": {"어부2": 0.8, "어부3": 0.6}}')


class FakeServer:
    """질문 종류(광장·어획·일기)에 맞는 답을 주는 openai 흉내. prompt_tokens 칸에 보낸 글자 수를 준다."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id="qwen2.5:7b")])

    def _create(self, **kw):
        system = kw["messages"][0]["content"]
        answer = SAY_ALL if "광장에 모였다" in system else DIARY if "일기를 쓴다" in system else FISH10
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=sum(len(m["content"]) for m in kw["messages"]),
                                  completion_tokens=20),
        )


def _llm_village(tmp_path, talk, days):
    config = make_config(talk=talk, llm_on=True)
    world = World.from_config(config)
    conn = open_db(tmp_path / f"v{talk}.db")
    logger = RunLogger.start(conn, "r", config_name="t", condition="llm", seed=1, model="qwen2.5:7b")
    caller = llm.LLMCaller(client=FakeServer(), meter=llm.CostMeter(1.0), record=logger.log_llm_call)
    result = run(world, build_llm_agents(config, caller, {i: "qwen2.5:7b" for i in IDS}), seed=1, days=days,
                 logger=logger)
    return conn, result


def test_llm_village_talks_then_fishes_then_writes_diary(tmp_path):
    conn, result = _llm_village(tmp_path, talk=True, days=2)
    assert result.end_reason == END_DAYS
    phases = conn.execute("SELECT phase, COUNT(*) FROM llm_calls GROUP BY phase ORDER BY phase").fetchall()
    assert phases == [("diary", 10), ("fish", 10), ("say", 10)]
    # 하루 안 순서: 광장 5 → 어획 5 → 일기 5
    day1 = [p for (p,) in conn.execute("SELECT phase FROM llm_calls WHERE day = 1 ORDER BY rowid")]
    assert day1 == ["say"] * 5 + ["fish"] * 5 + ["diary"] * 5
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (10,)
    (fish2,) = conn.execute("SELECT prompt FROM llm_calls WHERE day = 2 AND phase = 'fish' LIMIT 1").fetchone()
    assert "오늘 아침 광장에서 들은 말" in fish2 and "1일째 광장" in fish2 and "이웃 신뢰" in fish2
    conn.close()


def test_llm_village_with_talk_off_never_asks_to_talk(tmp_path):
    conn, _ = _llm_village(tmp_path, talk=False, days=2)
    assert conn.execute("SELECT COUNT(*) FROM llm_calls WHERE phase = 'say'").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
    (fish2,) = conn.execute("SELECT prompt FROM llm_calls WHERE day = 2 AND phase = 'fish' LIMIT 1").fetchone()
    # (가짜 일기 요약에는 "광장"이라는 낱말이 있어서, 광장 블록 이름으로 본다)
    for piece in ["광장에서 들은 말", "1일째 광장:", "이웃 신뢰"]:
        assert piece not in fish2, piece
    conn.close()


def test_prompt_length_still_stops_growing_with_talk(tmp_path):
    """대화가 붙어도 질문지 길이가 며칠 뒤 평평하다 (한 마디 200자 상한 + 최근 2일만)."""
    conn, result = _llm_village(tmp_path, talk=True, days=10)
    assert len(result.days) == 10
    for phase in ("say", "fish", "diary"):
        per_day = [m for (m,) in conn.execute(
            "SELECT MAX(prompt_tokens) FROM llm_calls WHERE phase = ? GROUP BY day ORDER BY day", (phase,)
        )]
        assert per_day[3] > per_day[0]
        settled = per_day[3:]
        # 말하는 순서가 날마다 바뀌어 "나/이름", "네가 처음으로" 줄만큼(약 15글자) 흔들린다. 늘어나지는 않는다
        assert max(settled) - min(settled) <= 30, (phase, per_day)
    conn.close()


def test_beliefs_from_diary_are_untouched_by_talk():
    world = World.from_config(make_config())
    world.get_agent("a1").beliefs = Beliefs(80, 0.5, 8)
    agents = talkers()
    run_day(world, agents, make_rng(1, "estimate"))
    assert "약 80마리" in agents["a1"].say_seen[0].memory
