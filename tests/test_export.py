"""export.py 테스트: 기록 읽기, 여러 DB 모으기, 이름 붙이기, HTML 끼워 넣기, CLI."""

import json
import sqlite3
from pathlib import Path

import pytest

from pondvillage import export as ex
from pondvillage import logger as lg
from pondvillage.agents.scripted import CooperativeBot, GreedyBot
from pondvillage.runner import run
from pondvillage.world import World, config_from_dict

RAW = {
    "pond": {
        "capacity": 100, "initial_stock": 100, "regen_multiplier": 2,
        "collapse_below": 5, "estimate_noise": [0.8, 1.2],
    },
    "economy": {
        "start_money": 20, "fish_price": 1, "daily_cost": 5,
        "max_catch_per_day": 20, "inspect_catch_factor": 0.5, "bankrupt_after_days": 3,
    },
    "run": {"days": 30, "seed": 1},
    "agents": [{"id": f"a{i}", "name": f"어부{i}", "persona": "어부"} for i in range(1, 6)],
}
CONFIG = config_from_dict(RAW)


def save_run(path: Path, run_id: str, bot_cls, condition: str, days: int) -> None:
    conn = lg.open_db(path)
    world = World.from_config(CONFIG)
    logger = lg.RunLogger.start(
        conn, run_id, config_name="test", condition=condition, seed=1, model="scripted",
        started_at="2026-11-15T00:00:00",
    )
    run(world, {a.id: bot_cls(a.id) for a in world.agents}, 1, days=days, logger=logger)
    conn.close()


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "runs"
    save_run(d / "greedy.db", "g1", GreedyBot, "greedy", 30)
    save_run(d / "coop.db", "c1", CooperativeBot, "coop", 3)
    return d


def test_list_runs_reads_status_and_day_count(runs_dir):
    conn = sqlite3.connect(runs_dir / "greedy.db")
    [info] = ex.list_runs(conn)
    conn.close()
    assert info["run_id"] == "g1"
    assert info["condition"] == "greedy"
    assert info["status"] == "collapsed"
    assert info["day_count"] == 1


def test_load_run_days_agents_and_money(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    data = ex.load_run(conn, "c1")
    conn.close()
    assert [d["day"] for d in data["days"]] == [1, 2, 3]
    day1 = data["days"][0]
    assert day1["stock_start"] == 100 and day1["total_catch"] == 50 and day1["stock_end"] == 100
    assert day1["collapsed"] is False
    assert [a["id"] for a in day1["agents"]] == [f"a{i}" for i in range(1, 6)]
    # 협력 봇: 10마리 팔고 생활비 5 → 하루 +5
    assert [a["money"] for a in data["days"][2]["agents"]] == [35] * 5
    assert all(a["catch"] == 10 and a["alive"] for a in day1["agents"])
    assert data["messages"] == []


def test_load_run_reads_messages(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    conn.execute(
        'INSERT INTO messages (run_id, day, "from", "to", text, is_rumor) VALUES (?, ?, ?, ?, ?, ?)',
        ("c1", 2, "a3", "a4", "연못이 곧 바닥난대", 1),
    )
    conn.commit()
    data = ex.load_run(conn, "c1")
    conn.close()
    assert data["messages"] == [{"day": 2, "from": "a3", "to": "a4", "text": "연못이 곧 바닥난대", "is_rumor": True}]


def test_collect_gathers_all_files_with_names(runs_dir):
    data = ex.collect(runs_dir, CONFIG)
    assert {(r["file"], r["run_id"]) for r in data["runs"]} == {("coop.db", "c1"), ("greedy.db", "g1")}
    # 실행 목록의 날 수와 날짜별 기록이 서로 덮어쓰지 않는다
    assert all(r["day_count"] == len(r["days"]) for r in data["runs"])
    assert data["agents"][0] == {"id": "a1", "name": "어부1"}
    assert data["economy"] == {"start_money": 20, "fish_price": 1, "daily_cost": 5}
    assert data["pond"] == {"capacity": 100, "collapse_below": 5}


def test_collect_skips_broken_files(runs_dir):
    (runs_dir / "broken.db").write_text("SQLite 아님", encoding="utf-8")
    sqlite3.connect(runs_dir / "empty.db").close()  # 표가 없는 빈 DB
    assert len(ex.collect(runs_dir, CONFIG)["runs"]) == 2


def test_collect_empty_folder(tmp_path):
    assert ex.collect(tmp_path / "없음", CONFIG)["runs"] == []


def test_render_html_puts_data_and_blocks_script_end():
    data = {"runs": [{"messages": [{"text": "</script><b>깨뜨리기"}]}]}
    html = ex.render_html(data, f"<script>const DATA={ex.DATA_MARK};</script>")
    assert ex.DATA_MARK not in html
    assert html.count("</script>") == 1  # 틀의 닫는 태그 하나만
    payload = html[len("<script>const DATA="):-len(";</script>")]
    assert json.loads(payload) == data


def test_render_html_needs_data_mark():
    with pytest.raises(ValueError):
        ex.render_html({}, "<html></html>")


def test_template_has_data_mark():
    assert ex.DATA_MARK in ex.TEMPLATE.read_text(encoding="utf-8")


def test_cli_writes_village_html(runs_dir, tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(RAW, ensure_ascii=False), encoding="utf-8")  # JSON은 YAML로도 읽힌다
    assert ex.main(["--runs-dir", str(runs_dir), "--config", str(cfg)]) == 0
    out = runs_dir / "village.html"
    html = out.read_text(encoding="utf-8")
    assert '"run_id": "g1"' in html and '"name": "어부1"' in html
    assert "실행 2개" in capsys.readouterr().out


def test_cli_no_runs(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(RAW, ensure_ascii=False), encoding="utf-8")
    assert ex.main(["--runs-dir", str(tmp_path / "runs"), "--config", str(cfg)]) == 1
    assert "기록이 없습니다" in capsys.readouterr().out


# ---------- 대시보드 v1 (P7): 모델·속마음·일기 ----------

def _call(conn, day, agent, phase, model, response=None):
    conn.execute(
        "INSERT INTO llm_calls (run_id, day, agent_id, phase, model, prompt_tokens, completion_tokens,"
        " latency_ms, cost_usd, prompt, response, parse_error) VALUES ('c1', ?, ?, ?, ?, 1, 1, 1, 0, '', ?, NULL)",
        (day, agent, phase, model, response),
    )


def _action(conn, day, phase, agent, body, reason=None, parse_error=None, clamped=0):
    conn.execute(
        "INSERT INTO actions (run_id, day, phase, agent_id, action_json, reason, parse_error, clamped)"
        " VALUES ('c1', ?, ?, ?, ?, ?, ?, ?)",
        (day, phase, agent, json.dumps(body, ensure_ascii=False), reason, parse_error, clamped),
    )


def test_bot_run_has_no_models_but_has_fish(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    data = ex.load_run(conn, "c1")
    conn.close()
    assert data["models"] == {}  # 봇은 LLM을 부르지 않는다 → 화면은 "협력 봇"으로 표시
    assert data["diaries"] == []
    day1 = [f for f in data["fish"] if f["day"] == 1]
    assert len(day1) == 5
    assert all(f["action"] == "fish" and f["amount"] == 10 and not f["inspected"] for f in day1)


def test_models_are_the_most_called_model_per_fisher(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    for _ in range(3):
        _call(conn, 1, "a1", "fish", "qwen2.5:7b")
    _call(conn, 1, "a1", "fish", "gemma3:4b")  # 섞여 있어도 가장 많이 부른 모델
    _call(conn, 1, "a2", "say", "gemma3:4b")
    assert ex.load_models(conn, "c1") == {"a1": "qwen2.5:7b", "a2": "gemma3:4b"}
    conn.close()


def test_fish_separates_inspect_then_nothing_from_plain_zero(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    conn.execute("DELETE FROM actions WHERE run_id = 'c1'")
    _action(conn, 1, "fish", "a1", {"action": "inspect_pond", "amount": 0}, "얼마나 남았나 보자")
    _action(conn, 1, "fish_after_inspect", "a1", {"action": "none", "amount": 0}, "오늘은 쉬자")
    _action(conn, 1, "fish", "a2", {"action": "none", "amount": 0}, None, parse_error="JSON 아님")
    _action(conn, 1, "fish", "a3", {"action": "fish", "amount": 7}, "적당히")
    fish = {f["agent"]: f for f in ex.load_fish(conn, "c1")}
    conn.close()
    assert fish["a1"] == {"day": 1, "agent": "a1", "inspected": True, "action": "none", "amount": 0,
                          "reason": "오늘은 쉬자", "parse_error": False}
    assert fish["a2"]["inspected"] is False and fish["a2"]["parse_error"] is True
    assert fish["a3"]["action"] == "fish" and fish["a3"]["amount"] == 7 and fish["a3"]["reason"] == "적당히"


def test_diaries_keep_full_text_only_when_summary_was_cut(runs_dir):
    conn = sqlite3.connect(runs_dir / "coop.db")
    long = "가" * 320
    beliefs = {"estimated_stock": 90, "confidence": 0.7, "sustainable_guess": 10,
               "heard_rumor": False, "believes_rumor": 0.0}
    _action(conn, 1, "diary", "a1", {"summary": long[:300], "beliefs": beliefs, "trust": {"a2": 0.8}}, clamped=1)
    _call(conn, 1, "a1", "diary", "qwen2.5:7b", json.dumps({"summary": long}, ensure_ascii=False))
    _action(conn, 1, "diary", "a2", {"summary": "짧은 일기", "beliefs": beliefs, "trust": {}}, clamped=1)  # 숫자만 맞춤
    _call(conn, 1, "a2", "diary", "qwen2.5:7b", json.dumps({"summary": "짧은 일기"}, ensure_ascii=False))
    _action(conn, 1, "diary", "a3", {}, parse_error="summary 없음")
    _call(conn, 1, "a3", "diary", "qwen2.5:7b", "깨진 답")
    d = {x["agent"]: x for x in ex.load_diaries(conn, "c1")}
    conn.close()
    assert d["a1"]["summary"] == long[:300] and d["a1"]["full"] == long and d["a1"]["clamped"] is True
    assert d["a1"]["trust"] == {"a2": 0.8} and d["a1"]["beliefs"]["estimated_stock"] == 90
    assert d["a2"]["clamped"] is True and d["a2"]["full"] is None
    assert d["a3"]["parse_error"] is True and d["a3"]["summary"] == "" and d["a3"]["full"] is None


def test_whisper_receiver_reaches_the_page(runs_dir, tmp_path):
    conn = sqlite3.connect(runs_dir / "coop.db")
    conn.execute(
        'INSERT INTO messages (run_id, day, "from", "to", text, is_rumor) VALUES (?, ?, ?, ?, ?, ?)',
        ("c1", 1, "a4", "a5", "나는 오늘 10마리만 잡을게", 0),
    )
    conn.commit()
    conn.close()
    run = next(r for r in ex.collect(runs_dir, CONFIG)["runs"] if r["run_id"] == "c1")
    assert run["messages"] == [{"day": 1, "from": "a4", "to": "a5", "text": "나는 오늘 10마리만 잡을게", "is_rumor": False}]


def test_template_has_no_code_hidden_behind_a_comment():
    """v0 버그 재발 막기: 줄 주석(//) 뒤에 변수 선언이 붙으면 브라우저가 그 선언을 읽지 않는다."""
    import re
    bad = [line for line in ex.TEMPLATE.read_text(encoding="utf-8").splitlines()
           if re.search(r"//[^\n]*(?:let|const|var)\s+[A-Za-z_$][\w$]*\s*=", line)]
    assert bad == []


def test_template_shows_theater_parts():
    html = ex.TEMPLATE.read_text(encoding="utf-8")
    for part in ("광장 극장", "두 마을 비교", 'data-tab="diary"', "timeline", "귓속말", "어제(", "modelOf"):
        assert part in html
