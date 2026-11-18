"""P8 실험 러너·분석 테스트: 자리 돌리기, 혼자 말하기, 모델별 묶어 부르기, 이어 돌리기, 지표 계산."""

import copy
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from pondvillage import analyze, experiment, llm, memory
from pondvillage.actions import TO_ALL, TO_SELF
from pondvillage.agents.base import Observation
from pondvillage.agents.llm_agent import build_llm_agents, fish_user_prompt, say_system_prompt
from pondvillage.runner import by_model, make_rng, run_day
from pondvillage.world import ConfigError, World, config_from_dict

ROOT = Path(__file__).resolve().parent.parent
Q, G = "qwen2.5:7b", "gemma3:4b"

RAW = {
    "pond": {"capacity": 100, "initial_stock": 100, "regen_multiplier": 2, "collapse_below": 5,
             "estimate_noise": [0.8, 1.2]},
    "economy": {"start_money": 20, "fish_price": 1, "daily_cost": 5, "max_catch_per_day": 20,
                "inspect_catch_factor": 0.5, "bankrupt_after_days": 3},
    "run": {"days": 30, "seed": 1},
    "talk": {"enabled": True, "max_chars": 200},
    "llm": {"base_url": "http://localhost:11434/v1", "model": Q, "cost_cap_usd": 1.0},
    "agents": [{"id": f"a{i}", "name": f"어부{i}", "persona": "어부"} for i in range(1, 6)],
}

SAY = '{"action": "say", "to": "어부2", "message": "오늘은 8마리만 잡겠다", "reason": "r"}'
FISH = '{"action": "fish", "amount": 8, "reason": "r"}'
DIARY = ('{"summary": "조금씩 잡았다.", '
         '"beliefs": {"estimated_stock": 90, "confidence": 0.7, "sustainable_guess": 12}, "trust": {}}')


class RoutingFake:
    """질문 종류(광장·어획·일기)에 맞는 답을 주는 가짜 Ollama. 부른 모델 순서와 질문을 기록한다."""

    def __init__(self, models=(Q, G), fish=FISH):
        self.fish = fish
        self.asked = []  # (모델, 종류, 질문 글)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in models])

    def _create(self, **kw):
        system = kw["messages"][0]["content"]
        user = kw["messages"][-1]["content"]
        if "일기를 쓴다" in system:
            kind, answer = "diary", DIARY
        elif '"action": "say"' in system:
            kind, answer = "say", SAY
        else:
            kind, answer = "fish", self.fish
        self.asked.append((kw["model"], kind, user))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20),
        )


def make_config(**talk):
    raw = copy.deepcopy(RAW)
    raw["talk"].update(talk)
    return config_from_dict(raw)


def make_village(config, models, fake):
    world = World.from_config(config)
    for a in world.agents:
        a.model = models[a.id]
    caller = llm.LLMCaller(config.llm.base_url, 0.7, client=fake)
    return world, build_llm_agents(config, caller, models)


# ---------------------------------------------------------------- 자리 돌리기 · 실험 파일


def test_rotate_models_shifts_one_seat_per_seed():
    x = (Q, G, Q, G, Q)
    assert experiment.rotate_models(x, 1) == [Q, G, Q, G, Q]
    assert experiment.rotate_models(x, 2) == [Q, Q, G, Q, G]
    assert experiment.rotate_models(x, 3) == [G, Q, Q, G, Q]
    # 한 명만 다른 마을: 그 한 명의 자리가 시드마다 한 칸씩 옮겨 간다
    o = (G, Q, Q, Q, Q)
    assert [experiment.rotate_models(o, s).index(G) for s in (1, 2, 3)] == [0, 1, 2]
    # 모델 수는 그대로
    for s in range(1, 8):
        assert sorted(experiment.rotate_models(x, s)) == sorted(x)


def test_repo_experiment_files_plan_21_runs_with_spec_numbers_untouched():
    files = sorted(ROOT.glob("configs/exp1_*.yaml"))
    planned = experiment.plan([experiment.load_experiment(f) for f in files])
    assert len(planned) == 21  # 마을 7종 × 시드 3
    conds = {p.condition for p in planned}
    assert conds == {"exp1-S-qwen", "exp1-S-gemma", "exp1-X-qwen3gemma2", "exp1-O-qwen4gemma1",
                     "exp1-O-gemma4qwen1", "exp1-solo-qwen", "exp1-off-qwen"}
    base = config_from_dict(experiment.load_experiment(files[0]).base_raw)
    for p in planned:
        assert p.days == 10 and p.seed in (1, 2, 3)
        assert p.config.pond == base.pond and p.config.economy == base.economy  # 숫자 규칙은 그대로
        assert not p.config.llm.persona_enabled
    solo = next(p for p in planned if p.condition == "exp1-solo-qwen")
    off = next(p for p in planned if p.condition == "exp1-off-qwen")
    assert solo.config.talk.solo and not off.config.talk.enabled
    x2 = next(p for p in planned if p.condition == "exp1-X-qwen3gemma2" and p.seed == 2)
    assert [x2.models[s.id] for s in x2.config.agents] == [Q, Q, G, Q, G]


def test_experiment_file_with_wrong_number_of_models_is_refused(tmp_path):
    (tmp_path / "base.yaml").write_text((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    bad = tmp_path / "exp1_bad.yaml"
    bad.write_text("experiment: exp1\nbase: base.yaml\ndays: 2\nseeds: [1]\n"
                   "villages:\n  - name: S-q\n    models: [a, b]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        experiment.load_experiment(bad)


def test_unknown_talk_mode_is_refused():
    with pytest.raises(ConfigError):
        make_config(mode="shout")


# ---------------------------------------------------------------- 혼자 말하기 (talk.mode = solo)


def test_solo_village_nobody_hears_anyone_but_self():
    config = make_config(mode="solo")
    fake = RoutingFake()
    world, agents = make_village(config, {f"a{i}": Q for i in range(1, 6)}, fake)
    day = run_day(world, agents, make_rng(1, "estimate"), None, make_rng(1, "talk_order"))
    assert len(day.says) == 5
    assert all(s.decision.to == TO_SELF for s in day.says)  # 어부2에게 귓속말을 해도 혼잣말로
    assert not any(s.decision.whisper for s in day.says)
    says = [u for m, k, u in fake.asked if k == "say"]
    fishes = [u for m, k, u in fake.asked if k == "fish"]
    # 광장 질문지에 남의 말이 없다, 어획 질문지에는 내 혼잣말 한 줄만
    assert not any("들은 말" in u for u in says)
    for u in fishes:
        assert "[오늘 아침 내가 적은 혼잣말]" in u and u.count("오늘은 8마리만 잡겠다") == 1
        assert "나(혼잣말, 아무도 듣지 않음)" in u
    # 기억에도 자기 말만
    for a in world.agents:
        plaza = [m.text for m in a.memory if "광장" in m.text]
        assert len(plaza) == 1 and plaza[0].count("오늘은 8마리만") == 1


def test_solo_prompt_says_nobody_listens_and_plaza_prompt_is_unchanged():
    solo = say_system_prompt(make_config(mode="solo"))
    plaza = say_system_prompt(make_config())
    assert "아무도 듣지 않" in solo and "귓속말" not in solo
    assert "귓속말" in plaza and "혼잣말" not in plaza
    obs = Observation(day=1, agent_id="a1", money=20, stock_estimate=100, yesterday_total_catch=None,
                      memory="", heard="나(모두에게): \"안녕\"")
    assert "[오늘 아침 광장에서 들은 말]" in fish_user_prompt("어부1", None, obs, 10)
    assert "[오늘 아침 내가 적은 혼잣말]" in fish_user_prompt("어부1", None, obs, 10, solo=True)


def test_plaza_message_to_self_is_heard_only_by_speaker():
    m = memory.PlazaMessage("a1", TO_SELF, "음")
    assert m.heard_by("a1") and not m.heard_by("a2")
    assert memory.PlazaMessage("a1", TO_ALL, "음").heard_by("a2")


# ---------------------------------------------------------------- 모델별 묶어 부르기


def test_fish_and_diary_calls_are_grouped_by_model_but_plaza_is_not():
    config = make_config()
    models = {"a1": Q, "a2": G, "a3": Q, "a4": G, "a5": Q}
    fake = RoutingFake()
    world, agents = make_village(config, models, fake)
    day = run_day(world, agents, make_rng(1, "estimate"), None, make_rng(1, "talk_order"))
    fish_models = [m for m, k, _ in fake.asked if k == "fish"]
    diary_models = [m for m, k, _ in fake.asked if k == "diary"]
    assert fish_models == [Q, Q, Q, G, G] and diary_models == [Q, Q, Q, G, G]
    # 결과·기록 순서는 마을 순서
    assert [t.agent_id for t in day.turns] == ["a1", "a2", "a3", "a4", "a5"]
    assert [d.agent_id for d in day.diaries] == ["a1", "a2", "a3", "a4", "a5"]


def test_by_model_keeps_village_order_within_each_model():
    world = World.from_config(make_config())
    for a, m in zip(world.agents, [G, Q, G, Q, Q]):
        a.model = m
    assert [a.id for a in by_model(world.agents)] == ["a1", "a3", "a2", "a4", "a5"]


# ---------------------------------------------------------------- 실험 러너 끝까지 (가짜 모델)


def _write_exp(tmp_path, days=2):
    (tmp_path / "base.yaml").write_text((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / "exp1_t.yaml"
    path.write_text(
        f"experiment: exp1\nbase: base.yaml\ndays: {days}\nseeds: [1, 2]\nvillages:\n"
        f"  - name: S-qwen\n    models: [{Q}, {Q}, {Q}, {Q}, {Q}]\n"
        f"  - name: O-qwen4gemma1\n    models: [{G}, {Q}, {Q}, {Q}, {Q}]\n    rotate: true\n"
        f"  - name: solo-qwen\n    models: [{Q}, {Q}, {Q}, {Q}, {Q}]\n    talk: {{mode: solo}}\n",
        encoding="utf-8",
    )
    return path


def test_experiment_runs_all_then_skips_finished_and_analysis_reads_them(tmp_path, monkeypatch, capsys):
    exp = _write_exp(tmp_path)
    runs = tmp_path / "runs"
    fake = RoutingFake()
    monkeypatch.setattr(llm, "make_client", lambda *a, **kw: fake)
    assert experiment.main([str(exp), "--runs-dir", str(runs)]) == 0
    dbs = sorted(runs.glob("*.db"))
    assert len(dbs) == 6
    conn = sqlite3.connect(dbs[0])
    assert conn.execute("SELECT config_name FROM runs").fetchone() == ("exp1_t",)
    conn.close()
    calls = len(fake.asked)

    # 다시 돌리면 모두 건너뛴다 (호출 0번)
    assert experiment.main([str(exp), "--runs-dir", str(runs)]) == 0
    assert len(fake.asked) == calls
    assert "이미 끝남 6번 · 남음 0번" in capsys.readouterr().out

    data = analyze.collect_runs(runs, "exp1")
    assert {(r.village, r.seed) for r in data} == {(v, s) for v in ("S-qwen", "O-qwen4gemma1", "solo-qwen")
                                                   for s in (1, 2)}
    o2 = next(r for r in data if r.village == "O-qwen4gemma1" and r.seed == 2)
    assert o2.models["agent_02"] == G  # 시드 2: 다른 한 명은 순자 자리
    solo = next(r for r in data if r.village == "solo-qwen")
    assert solo.messages and all(m["to"] == TO_SELF for m in solo.messages)

    stats = analyze.model_stats(data)
    assert stats[(Q, "S")].avg_request == 8 and stats[(G, "O-한명")].agent_days == 4
    assert stats[(Q, "O-다수")].kept == stats[(Q, "O-다수")].num_says > 0  # "8마리만" 말하고 8 요청
    assert stats[(Q, "S")].guesses_first and stats[(Q, "S")].guesses_last[0] == 12

    assert analyze.main(["--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "| 마을 |" in out and "S-qwen" in out and "섞임 효과" in out
    report = (runs / "exp1_report.html").read_text(encoding="utf-8")
    assert "<svg" in report and "O-qwen4gemma1" in report and "</html>" in report


def test_unfinished_run_is_done_again(tmp_path, monkeypatch):
    exp = _write_exp(tmp_path, days=1)
    runs = tmp_path / "runs"
    monkeypatch.setattr(llm, "make_client", lambda *a, **kw: RoutingFake())
    assert experiment.main([str(exp), "--runs-dir", str(runs)]) == 0
    db = next(iter(sorted(runs.glob("*S-qwen_s1.db"))))
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET status = 'running'")  # 중간에 끊긴 것처럼
    conn.commit()
    conn.close()
    done = experiment.finished_runs(runs)
    assert ("exp1-S-qwen", 1) not in done and ("exp1-S-qwen", 2) in done


def test_experiment_dry_run_calls_nothing(tmp_path, monkeypatch, capsys):
    exp = _write_exp(tmp_path)

    def boom(*a, **kw):
        raise AssertionError("dry-run에서 모델을 부르면 안 된다")

    monkeypatch.setattr(llm, "check_ollama", boom)
    assert experiment.main([str(exp), "--runs-dir", str(tmp_path / "runs"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "남음 6번" in out and "혼자 말하기" in out
    assert not list((tmp_path / "runs").glob("*.db"))


def test_experiment_stops_when_ollama_is_down(tmp_path, monkeypatch, capsys):
    exp = _write_exp(tmp_path)

    def down(*a, **kw):
        raise llm.LLMUnavailable("WSL을 깨워 주세요")

    monkeypatch.setattr(llm, "check_ollama", down)
    assert experiment.main([str(exp), "--runs-dir", str(tmp_path / "runs")]) == 1
    assert "WSL" in capsys.readouterr().out


# ---------------------------------------------------------------- 지표 계산


def test_gini():
    assert analyze.gini([10, 10, 10, 10, 10]) == 0
    assert analyze.gini([0, 0, 0, 0, 0]) == 0
    assert analyze.gini([0, 0, 0, 0, 50]) == pytest.approx(0.8)


def _run(village="S-qwen", days=None, fish=None, models=None, messages=(), guesses=()):
    days = days or [{"day": 1, "stock_start": 100, "stock_end": 100, "total_catch": 50, "collapsed": False}]
    return analyze.RunData(
        "r", village, 1, "days", days, models or {f"a{i}": Q for i in range(1, 6)}, fish or [], {"a1": 50},
        list(guesses), list(messages),
    )


def test_run_metrics_and_stock_series_after_collapse():
    days = [
        {"day": 1, "stock_start": 100, "stock_end": 40, "total_catch": 80, "collapsed": False},
        {"day": 2, "stock_start": 40, "stock_end": 2, "total_catch": 38, "collapsed": True},
    ]
    fish = [{"day": 1, "agent": f"a{i}", "inspected": False, "amount": 16, "parse_error": False} for i in range(1, 6)]
    r = _run(days=days, fish=fish)
    m = analyze.run_metrics(r)
    assert (m.days, m.collapsed, m.day1_request, m.total_catch) == (2, True, 80, 118)
    assert m.avg_stock == 21 and m.efficiency == pytest.approx(118 / 100)
    assert analyze.stock_series(r, 4) == [40, 2, 0, 0]


def test_role_in_oneoff_village():
    r = _run(village="O-qwen4gemma1", models={"a1": G, "a2": Q, "a3": Q, "a4": Q, "a5": Q})
    assert analyze.role(r, "a1") == "O-한명" and analyze.role(r, "a2") == "O-다수"
    assert analyze.role(_run(village="X-qg"), "a1") == "X"


def test_promise_counts_requests_at_or_below_the_number_said():
    fish = [{"day": 1, "agent": "a1", "inspected": False, "amount": 12, "parse_error": False},
            {"day": 1, "agent": "a2", "inspected": True, "amount": 5, "parse_error": False}]
    msgs = [{"day": 1, "from": "a1", "to": "all", "text": "각자 10마리씩"},
            {"day": 1, "from": "a2", "to": "all", "text": "10마리 이하로"},
            {"day": 1, "from": "a3", "to": "all", "text": "적당히"}]
    s = analyze.model_stats([_run(fish=fish, messages=msgs)])[(Q, "S")]
    assert (s.says, s.num_says, s.kept, s.inspects, s.agent_days) == (3, 2, 1, 1, 2)


def test_mixing_effect_is_mixed_minus_same():
    stats = {(Q, "S"): analyze.ModelStats(agent_days=10, request_sum=80),
             (Q, "X"): analyze.ModelStats(agent_days=10, request_sum=100)}
    (row,) = analyze.mixing_effect(stats)
    assert (row["place"], row["S"], row["mixed"], row["diff"]) == ("X", 8, 10, 2)


def test_analysis_ignores_other_experiments_and_unfinished_runs(tmp_path):
    from pondvillage.logger import RunLogger, open_db
    conn = open_db(tmp_path / "x.db")
    RunLogger.start(conn, "a", config_name="base", condition="llm-same", seed=1, model=Q).finish("days")
    RunLogger.start(conn, "b", config_name="e", condition="exp1-S-qwen", seed=1, model=Q)  # running
    conn.close()
    assert analyze.collect_runs(tmp_path, "exp1") == []


# ---------------------------------------------------------------- P9 준비: 공정 비교·시드 늘리기


def _fish(day, agent, amount):
    return {"day": day, "agent": agent, "inspected": False, "amount": amount, "parse_error": False}


def test_early_requests_keep_days_apart_and_drop_day_four_on():
    # 붕괴한 마을: 1일째 20씩. 버틴 마을: 1일째 10, 2~5일째 2. 전체 평균은 버틴 마을이 유리하지만 같은 날끼리 보면 다르다.
    crashed = _run(fish=[_fish(1, f"a{i}", 20) for i in range(1, 6)])
    lasted = _run(fish=[_fish(d, f"a{i}", 10 if d == 1 else 2) for d in range(1, 6) for i in range(1, 6)])
    early = analyze.early_requests([crashed, lasted])
    day1, day2, day3 = early[(Q, "S")]
    assert (len(day1), sum(day1) / len(day1)) == (10, 15)
    assert (len(day2), len(day3)) == (5, 5)  # 2일째부터는 버틴 마을만 (n이 줄어든 게 보인다)
    head, row = analyze.early_table(early)
    assert row[2:] == ["15.0 (10)", "2.0 (5)", "2.0 (5)", "8.5 (20)"]


def test_seed_stability_uses_first_k_seeds_only():
    def m(seed, day1, collapsed):
        return analyze.RunMetrics("S-qwen", "S", seed, 3, collapsed, 0, 0, 0, day1, 0)
    ms = [m(s, 50 + 10 * (s - 1), s % 2 == 1) for s in range(1, 6)]  # 50 60 70 80 90
    rows = analyze.seed_stability(ms)
    assert [r["k"] for r in rows] == [3, 5]  # 10개 칸은 시드가 모자라 없음
    assert rows[0]["mean"] == 60 and rows[0]["sd"] == pytest.approx(10) and rows[0]["collapsed"] == 2
    assert rows[1]["mean"] == 70 and rows[1]["collapsed"] == 3


def test_parse_seeds():
    assert experiment.parse_seeds("1-10") == tuple(range(1, 11))
    assert experiment.parse_seeds("1,2,5") == (1, 2, 5)
    assert experiment.parse_seeds("1-3,2,7") == (1, 2, 3, 7)
    for bad in ("", "a", "0-3", "3-x"):
        with pytest.raises(ConfigError):
            experiment.parse_seeds(bad)


def test_seeds_option_overrides_file_without_touching_it(tmp_path, capsys):
    path = _write_exp(tmp_path)
    before = Path(path).read_text(encoding="utf-8")
    exp = experiment.load_experiment(path)
    planned = experiment.plan([exp], (4, 5))
    assert sorted({p.seed for p in planned}) == [4, 5]
    assert experiment.main([str(path), "--runs-dir", str(tmp_path / "r"), "--seeds", "1-4", "--dry-run"]) == 0
    assert "시드 4" in capsys.readouterr().out
    assert Path(path).read_text(encoding="utf-8") == before
    assert experiment.main([str(path), "--seeds", "x", "--dry-run"]) == 1
