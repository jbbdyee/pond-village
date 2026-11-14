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
