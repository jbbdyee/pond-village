"""SQLite 기록.

실행 결과를 `runs/*.db` 파일 하나(SQLite)에 남긴다. 표 모양은 SPEC 7장 그대로다.
- 채우는 표: runs(실행 소개), days(날마다 연못), agent_days(날마다 어부), actions(어부가 고른 행동),
  messages(광장에서 나온 말, P6. to는 "all" 또는 받는 어부 id),
  llm_calls(LLM 호출 한 번마다: 보낸 질문·받은 답·토큰·시간·비용·해석 실패)
- 빈 표만 만들어 두는 것: votes(3단계 투표)

기록원은 판정하지 않는다. 러너가 넘겨준 하루 결과(`DayResult`)를 표 칸에 옮겨 적기만 한다.
하루가 끝날 때마다 저장을 확정(commit)하므로, 중간에 멈춰도 그날까지는 남는다.

저장된 내용 보기:
    uv run python -m pondvillage.logger runs/<파일>.db            # 실행 목록 (하나뿐이면 날짜별 표까지)
    uv run python -m pondvillage.logger runs/<파일>.db --run <run_id>
    uv run python -m pondvillage.logger runs/<파일>.db --tokens   # 날짜별 질문지 길이(프롬프트 토큰)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # runner가 logger를 부르므로, 서로 부르는 고리를 피한다
    from pondvillage.llm import CallRecord
    from pondvillage.runner import DayResult
    from pondvillage.world import World

STATUS_RUNNING = "running"  # 진행 중. 끝나면 끝난 이유(collapsed / days / all_left)로 바뀐다
STATUS_ERROR = "error"  # 중간에 오류로 멈춤 (LLM 서버에 닿지 않을 때 포함)
STATUS_ABORTED_COST = "aborted_cost"  # 비용 상한을 넘어 멈춤 (SPEC 8장)

PHASE_FISH = "fish"  # 어획 페이즈 첫 답
PHASE_FISH_AFTER_INSPECT = "fish_after_inspect"  # 연못을 확인한 뒤 다시 고른 답
PHASE_DIARY = "diary"  # 저녁 일기 (P5)
PHASE_SAY = "say"  # 아침 광장 한 마디 (P6)

# SPEC 7장. messages의 from/to는 SQL 예약어라 따옴표로 감싼다.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config_name TEXT NOT NULL,
    condition TEXT NOT NULL,
    seed INTEGER NOT NULL,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS days (
    run_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    stock_start INTEGER NOT NULL,
    stock_end INTEGER NOT NULL,
    total_catch INTEGER NOT NULL,
    collapsed INTEGER NOT NULL,
    PRIMARY KEY (run_id, day)
);
CREATE TABLE IF NOT EXISTS agent_days (
    run_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    money INTEGER NOT NULL,
    alive INTEGER NOT NULL,
    catch INTEGER NOT NULL,
    violated INTEGER NOT NULL,
    beliefs_json TEXT,
    trust_json TEXT,
    PRIMARY KEY (run_id, day, agent_id)
);
CREATE TABLE IF NOT EXISTS messages (
    run_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    "from" TEXT NOT NULL,
    "to" TEXT,
    text TEXT NOT NULL,
    is_rumor INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS actions (
    run_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    phase TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    action_json TEXT NOT NULL,
    reason TEXT,
    parse_error TEXT,
    clamped INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS votes (
    run_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    proposal_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    vote TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL,
    day INTEGER,
    agent_id TEXT,
    phase TEXT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,
    prompt TEXT,
    response TEXT,
    parse_error TEXT
);
"""

TABLES = ("runs", "days", "agent_days", "messages", "actions", "votes", "llm_calls")


def open_db(path: str | Path) -> sqlite3.Connection:
    """DB 파일을 열고(없으면 폴더와 함께 만들고) 표 7개를 준비한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def make_stamp(now: datetime | None = None) -> str:
    """파일·실행 이름 앞에 붙는 시각. 예: 20261114-153000"""
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def make_run_id(stamp: str, condition: str, seed: int) -> str:
    """실행 이름. 예: 20261114-153000_coop_s1"""
    return f"{stamp}_{condition}_s{seed}"


class RunLogger:
    """실행 한 번의 기록원. `start`로 만들고, 매일 `log_day`, 끝나면 `finish`."""

    def __init__(self, conn: sqlite3.Connection, run_id: str) -> None:
        self.conn = conn
        self.run_id = run_id

    @classmethod
    def start(
        cls,
        conn: sqlite3.Connection,
        run_id: str,
        *,
        config_name: str,
        condition: str,
        seed: int,
        model: str,
        started_at: str | None = None,
    ) -> RunLogger:
        started_at = started_at or datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO runs (run_id, config_name, condition, seed, model, started_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, config_name, condition, seed, model, started_at, STATUS_RUNNING),
        )
        conn.commit()
        return cls(conn, run_id)

    def log_day(self, result: DayResult, world: World) -> None:
        """하루 결과를 days·agent_days·actions에 옮겨 적고 저장을 확정한다.

        world는 정산이 끝난 뒤의 마을이다. 믿음·신뢰(아직 없음)를 읽으려고 받는다.
        """
        rid, day = self.run_id, result.day
        self.conn.execute(
            "INSERT INTO days (run_id, day, stock_start, stock_end, total_catch, collapsed)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (rid, day, result.stock_start, result.stock_end, result.total_catch, int(result.collapsed)),
        )
        # 떠난 어부도 남긴다 (화면에서 "떠남"으로 보이게). 규칙 투표(3단계) 전에는 어길 규칙이 없어 violated = 0.
        for agent in world.agents:
            beliefs = json.dumps(asdict(agent.beliefs), ensure_ascii=False) if agent.beliefs else None
            trust = json.dumps(agent.trust, ensure_ascii=False) if agent.trust else None
            self.conn.execute(
                "INSERT INTO agent_days (run_id, day, agent_id, money, alive, catch, violated,"
                " beliefs_json, trust_json) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    rid, day, agent.id, result.money[agent.id], int(result.alive[agent.id]),
                    result.catches.get(agent.id, 0), beliefs, trust,
                ),
            )
        # 광장: 말한 순서대로. 해석 실패(말 안 함)는 actions에만 남고 messages에는 없다.
        for say in result.says:
            d = say.decision
            if not d.silent:
                self.conn.execute(
                    'INSERT INTO messages (run_id, day, "from", "to", text, is_rumor) VALUES (?, ?, ?, ?, ?, 0)',
                    (rid, day, say.agent_id, d.to, d.message),
                )
            self.conn.execute(
                "INSERT INTO actions (run_id, day, phase, agent_id, action_json, reason, parse_error, clamped)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rid, day, PHASE_SAY, say.agent_id,
                    json.dumps({} if d.silent else {"action": "say", "to": d.to, "message": d.message},
                               ensure_ascii=False),
                    d.reason, d.parse_error, int(d.clamped),
                ),
            )
        for turn in result.turns:
            answers = [(PHASE_FISH, turn.first), (PHASE_FISH_AFTER_INSPECT, turn.decision)] if turn.inspected \
                else [(PHASE_FISH, turn.decision)]
            for phase, d in answers:
                self.conn.execute(
                    "INSERT INTO actions (run_id, day, phase, agent_id, action_json, reason, parse_error, clamped)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rid, day, phase, turn.agent_id,
                        json.dumps({"action": d.kind, "amount": d.amount}, ensure_ascii=False),
                        d.reason, d.parse_error, int(d.clamped),
                    ),
                )
        # 일기: 검사한 내용을 action_json에 (실패면 빈 dict + parse_error). 믿음·신뢰는 위 agent_days에도 있다.
        for entry in result.diaries:
            body = {} if entry.parse_error else {
                "summary": entry.summary,
                "beliefs": asdict(entry.beliefs) if entry.beliefs else None,
                "trust": entry.trust,
            }
            self.conn.execute(
                "INSERT INTO actions (run_id, day, phase, agent_id, action_json, reason, parse_error, clamped)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                (rid, day, PHASE_DIARY, entry.agent_id, json.dumps(body, ensure_ascii=False),
                 entry.parse_error, int(entry.clamped)),
            )
        self.conn.commit()

    def log_llm_call(self, rec: CallRecord) -> None:
        """LLM 호출 한 번을 llm_calls에 적는다. 저장 확정은 그날 저녁(log_day)이나 finish에서."""
        self.conn.execute(
            "INSERT INTO llm_calls (run_id, day, agent_id, phase, model, prompt_tokens, completion_tokens,"
            " latency_ms, cost_usd, prompt, response, parse_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.run_id, rec.day, rec.agent_id, rec.phase, rec.model, rec.prompt_tokens,
                rec.completion_tokens, rec.latency_ms, rec.cost_usd, rec.prompt, rec.response, rec.parse_error,
            ),
        )

    def finish(self, status: str) -> None:
        """끝난 이유(또는 error)를 runs.status에 적는다."""
        self.conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, self.run_id))
        self.conn.commit()


# ---------------------------------------------------------------- 저장된 내용 보기 (콘솔)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(len(rows[0]))]
    return "\n".join("".join(cell.rjust(w) for cell, w in zip(r, widths)) for r in rows)


def format_runs(conn: sqlite3.Connection) -> str:
    rows = [["run_id", "봇/조건", "시드", "상태", "일수", "시작"]]
    for run_id, condition, seed, status, started_at, ndays in conn.execute(
        "SELECT r.run_id, r.condition, r.seed, r.status, r.started_at,"
        " (SELECT COUNT(*) FROM days d WHERE d.run_id = r.run_id)"
        " FROM runs r ORDER BY r.rowid"
    ):
        rows.append([run_id, condition, str(seed), status, str(ndays), started_at])
    return _table(rows)


def format_run_days(conn: sqlite3.Connection, run_id: str) -> str:
    """날짜별 아침·저녁 재고, 총어획과 어부별 돈 (떠난 어부는 "떠남(돈)")."""
    agents = [a for (a,) in conn.execute(
        "SELECT DISTINCT agent_id FROM agent_days WHERE run_id = ? ORDER BY agent_id", (run_id,)
    )]
    money: dict[tuple[int, str], str] = {}
    for day, agent_id, m, alive in conn.execute(
        "SELECT day, agent_id, money, alive FROM agent_days WHERE run_id = ?", (run_id,)
    ):
        money[(day, agent_id)] = str(m) if alive else f"떠남({m})"
    rows = [["날", "아침재고", "총어획", "저녁재고"] + agents]
    for day, s0, total, s1, collapsed in conn.execute(
        "SELECT day, stock_start, total_catch, stock_end, collapsed FROM days WHERE run_id = ? ORDER BY day",
        (run_id,),
    ):
        rows.append(
            [str(day), str(s0), str(total), f"{s1} 붕괴" if collapsed else str(s1)]
            + [money.get((day, a), "") for a in agents]
        )
    return _table(rows)


TOKEN_PHASES = ((PHASE_SAY, "광장"), (PHASE_FISH, "어획"), (PHASE_DIARY, "일기"))


def format_tokens(rows: list[tuple[int, str, int | None]]) -> str:
    """(날, 단계, 프롬프트 토큰) 목록 → 날짜별 단계별 평균·최대 표.

    P5 완료 기준 확인용: 기억이 붙어도 질문지 길이가 며칠 뒤 평평해져야 한다.
    해석 실패로 다시 물은 호출(앞 답이 붙어 길다)과 연못 확인 뒤 다시 물은 호출은 빼고 첫 질문만 본다.
    """
    by: dict[tuple[int, str], list[int]] = {}
    for day, phase, tokens in rows:
        if tokens is None:
            continue
        by.setdefault((day, phase), []).append(tokens)
    days = sorted({d for d, _ in by})
    # 기록이 하나도 없는 단계(예: 대화를 끈 마을의 광장)는 칸을 만들지 않는다
    phases = [(p, label) for p, label in TOKEN_PHASES if any(ph == p for _, ph in by)]
    header = ["날"]
    for _, label in phases:
        header += [f"{label} 평균", f"{label} 최대"]
    table = [header]
    for d in days:
        row = [str(d)]
        for phase, _ in phases:
            vals = by.get((d, phase))
            row += [f"{sum(vals) / len(vals):.0f}", str(max(vals))] if vals else ["-", "-"]
        table.append(row)
    return _table(table)


def token_rows(conn: sqlite3.Connection, run_id: str) -> list[tuple[int, str, int | None]]:
    """llm_calls에서 첫 질문만 고른다. 다시 물은 호출은 보낸 질문에 앞 답(assistant)이 들어 있다."""
    rows = []
    for day, phase, tokens, prompt in conn.execute(
        "SELECT day, phase, prompt_tokens, prompt FROM llm_calls WHERE run_id = ? ORDER BY rowid", (run_id,)
    ):
        try:
            roles = [m.get("role") for m in json.loads(prompt or "[]")]
        except (ValueError, AttributeError):
            roles = []
        if "assistant" not in roles:
            rows.append((day, phase, tokens))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="runs/*.db에 저장된 실행을 콘솔 표로 본다.")
    parser.add_argument("db", help="DB 파일 경로 (예: runs/20261114-153000_coop_s1.db)")
    parser.add_argument("--run", default=None, help="날짜별 표를 볼 run_id")
    parser.add_argument("--tokens", action="store_true", help="날짜별 질문지 길이(프롬프트 토큰) 표도 보인다")
    args = parser.parse_args(argv)

    path = Path(args.db)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return 1
    conn = sqlite3.connect(path)
    try:
        print(format_runs(conn))
        run_ids = [r for (r,) in conn.execute("SELECT run_id FROM runs ORDER BY rowid")]
        run_id = args.run or (run_ids[0] if len(run_ids) == 1 else None)
        if run_id is not None:
            if run_id not in run_ids:
                print(f"\n이 파일에 없는 run_id: {run_id}")
                return 1
            print(f"\n{run_id}")
            print(format_run_days(conn, run_id))
            if args.tokens:
                print("\n질문지 길이 (프롬프트 토큰, 첫 질문만)")
                print(format_tokens(token_rows(conn, run_id)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
