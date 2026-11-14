"""마을 미리보기 화면 만들기 (대시보드 v0).

runs 폴더의 실행 기록(`runs/*.db`)을 모두 읽어, 게임풍 화면 틀(`dashboard/village.html`)에
데이터를 끼워 넣은 HTML 파일 한 장을 만든다. 서버도 새 라이브러리도 필요 없고,
만든 파일을 브라우저로 열면 두 마을을 골라 나란히 재생·비교할 수 있다.

화면은 기록을 "보여 주기만" 한다. 어부가 걸어 다니는 길은 연출이고, 숫자(재고·잡은 수·돈·생존)는
전부 DB에서 온다. LLM 호출 없음, 비용 0원.

    uv run python -m pondvillage.export                       # runs/village.html 만들기
    uv run python -m pondvillage.export --out runs/my.html
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from pondvillage.world import Config, load_config

TEMPLATE = Path(__file__).resolve().parents[2] / "dashboard" / "village.html"
DATA_MARK = "/*__POND_DATA__*/null"  # 화면 틀에서 데이터가 들어갈 자리


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    """DB 파일 하나에 든 실행 목록 (기록 순서대로)."""
    return [
        {"run_id": r, "condition": c, "seed": s, "status": st, "started_at": t, "day_count": n}
        for r, c, s, st, t, n in conn.execute(
            "SELECT r.run_id, r.condition, r.seed, r.status, r.started_at,"
            " (SELECT COUNT(*) FROM days d WHERE d.run_id = r.run_id)"
            " FROM runs r ORDER BY r.rowid"
        )
    ]


def load_run(conn: sqlite3.Connection, run_id: str) -> dict:
    """실행 하나의 날짜별 연못·어부 기록과 광장 대화."""
    agents_by_day: dict[int, list[dict]] = {}
    for day, agent_id, money, alive, catch in conn.execute(
        "SELECT day, agent_id, money, alive, catch FROM agent_days WHERE run_id = ? ORDER BY day, agent_id",
        (run_id,),
    ):
        agents_by_day.setdefault(day, []).append(
            {"id": agent_id, "money": money, "alive": bool(alive), "catch": catch}
        )
    days = [
        {
            "day": day, "stock_start": s0, "stock_end": s1, "total_catch": total,
            "collapsed": bool(collapsed), "agents": agents_by_day.get(day, []),
        }
        for day, s0, s1, total, collapsed in conn.execute(
            "SELECT day, stock_start, stock_end, total_catch, collapsed FROM days WHERE run_id = ? ORDER BY day",
            (run_id,),
        )
    ]
    messages = [
        {"day": day, "from": frm, "to": to, "text": text, "is_rumor": bool(rumor)}
        for day, frm, to, text, rumor in conn.execute(
            'SELECT day, "from", "to", text, is_rumor FROM messages WHERE run_id = ? ORDER BY day, rowid',
            (run_id,),
        )
    ]
    return {"days": days, "messages": messages}


def collect(runs_dir: str | Path, config: Config) -> dict:
    """runs 폴더의 모든 DB에서 실행을 모아 화면용 데이터 하나로 만든다.

    SQLite가 아니거나 표가 없는 파일은 건너뛴다 (화면 만들기가 멈추지 않게).
    """
    runs: list[dict] = []
    for path in sorted(Path(runs_dir).glob("*.db")):
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            for info in list_runs(conn):
                runs.append({**info, "file": path.name, **load_run(conn, info["run_id"])})
        except sqlite3.Error:
            continue
        finally:
            conn.close()
    return {
        "agents": [{"id": a.id, "name": a.name} for a in config.agents],
        "economy": {
            "start_money": config.economy.start_money,
            "fish_price": config.economy.fish_price,
            "daily_cost": config.economy.daily_cost,
        },
        "pond": {"capacity": config.pond.capacity, "collapse_below": config.pond.collapse_below},
        "runs": runs,
    }


def render_html(data: dict, template: str) -> str:
    """화면 틀의 데이터 자리에 JSON을 넣는다. 대화 글에 `</script>`가 있어도 화면이 깨지지 않게 막는다."""
    if DATA_MARK not in template:
        raise ValueError(f"화면 틀에 데이터 자리({DATA_MARK})가 없습니다")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(DATA_MARK, payload, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="runs/*.db 기록으로 마을 미리보기 화면(HTML 한 장)을 만든다.")
    parser.add_argument("--runs-dir", default="runs", help="DB 파일이 있는 폴더 (기본 runs)")
    parser.add_argument("--config", default="configs/base.yaml", help="어부 이름을 읽을 설정 파일")
    parser.add_argument("--out", default=None, help="만들 HTML 경로 (기본 <runs-dir>/village.html)")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    data = collect(runs_dir, load_config(args.config))
    if not data["runs"]:
        print(f"기록이 없습니다: {runs_dir}/*.db")
        print("먼저 실행해 보세요: uv run python -m pondvillage.runner --baseline --seeds 10")
        return 1
    out = Path(args.out) if args.out else runs_dir / "village.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data, TEMPLATE.read_text(encoding="utf-8")), encoding="utf-8")
    files = len({r["file"] for r in data["runs"]})
    print(f"실행 {len(data['runs'])}개 (DB 파일 {files}개)를 담았습니다")
    print(f"화면: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
