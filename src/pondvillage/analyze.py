"""실험 분석: `runs/*.db`에서 실험 마을 기록을 모아 EXPERIMENT 1단계 지표 표와 그래프를 만든다 (P8).

- 콘솔: 마크다운 표 (NOTES·블로그에 붙여 넣기용)
- `runs/exp1_report.html`: 표 + 그래프 3장 (라이브러리 없이 SVG로 직접 그림, 서버 없음)

읽기만 한다. LLM 호출 없음, 비용 0원. 실험 기록은 runs.condition이 "<실험>-<마을>"인 것
(`experiment.py`가 적는다). 같은 마을·시드가 여러 번 끝났으면 가장 최근 것을 쓴다.

지표 정의 (EXPERIMENT 1단계 "측정 지표"):
- 진행 일수 / 끝: 붕괴한 날, 또는 정한 날까지 버팀
- 평균 잔존량: 날짜별 저녁 재고(stock_end) 평균
- 효율: 총 어획 ÷ (50 × 진행 일수)
- 불평등: 어부별 누적 어획의 지니 계수 (0 = 똑같이 나눔)
- 요청: 어부 1명 하루 최종 요청 마릿수 (확인 뒤 안 잡음·해석 실패는 0). 러너 끝 "모델별" 한 줄과 같은 정의
- 확인: 연못 확인(inspect_pond) 횟수 ÷ 어부-일
- 말: 발언 수, "N마리"가 들어간 말, 그 말을 한 날 요청이 말 속 가장 큰 N 이하였는지 (어림 지표, 아래 한계)
- 학습: 일기의 1인당 괜찮은 양(sustainable_guess)이 실제 값 10에 가까워지는지

    uv run python -m pondvillage.analyze                  # exp1 전부
    uv run python -m pondvillage.analyze --experiment exp1 --out runs/exp1_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pondvillage.export import load_fish, load_models

SUSTAINABLE_TOTAL = 50  # 마을 전체가 매일 잡아도 연못이 100으로 돌아오는 양 (SPEC 1장: 100 → 50 → 재생 100)
SUSTAINABLE_EACH = 10  # 5명이 나누면 1인 10
FINISHED = {"days", "collapsed", "all_left"}
NUM_FISH = re.compile(r"(\d+)\s*마리")

GROUP_LABELS = {
    "S": "같은 모델 마을",
    "X": "섞인 마을",
    "O": "한 명만 다른 마을",
    "solo": "혼자 말하기",
    "off": "대화 끈 마을",
}


# ---------------------------------------------------------------- 읽기


@dataclass
class RunData:
    """실험 마을 한 번의 기록 (분석에 필요한 것만)."""

    run_id: str
    village: str  # 예: S-qwen
    seed: int
    status: str
    days: list[dict]  # day, stock_start, stock_end, total_catch, collapsed
    models: dict[str, str]  # 어부 id → 모델
    fish: list[dict]  # day, agent, inspected, amount, parse_error
    catches: dict[str, int]  # 어부 id → 누적 어획
    guesses: list[tuple[int, str, int]]  # (날, 어부, sustainable_guess)
    messages: list[dict]  # day, from, to, text

    @property
    def group(self) -> str:
        return self.village.split("-", 1)[0]


def _read_run(conn: sqlite3.Connection, run_id: str, village: str, seed: int, status: str) -> RunData:
    days = [
        {"day": d, "stock_start": s0, "stock_end": s1, "total_catch": tc, "collapsed": bool(c)}
        for d, s0, s1, tc, c in conn.execute(
            "SELECT day, stock_start, stock_end, total_catch, collapsed FROM days WHERE run_id = ? ORDER BY day",
            (run_id,),
        )
    ]
    catches: dict[str, int] = defaultdict(int)
    guesses: list[tuple[int, str, int]] = []
    for day, agent, catch, beliefs in conn.execute(
        "SELECT day, agent_id, catch, beliefs_json FROM agent_days WHERE run_id = ? ORDER BY day, agent_id", (run_id,)
    ):
        catches[agent] += catch
        if beliefs:
            try:
                g = json.loads(beliefs).get("sustainable_guess")
            except (json.JSONDecodeError, AttributeError):
                g = None
            if isinstance(g, (int, float)):
                guesses.append((day, agent, int(g)))
    messages = [
        {"day": d, "from": f, "to": t, "text": x}
        for d, f, t, x in conn.execute(
            'SELECT day, "from", "to", text FROM messages WHERE run_id = ? ORDER BY day, rowid', (run_id,)
        )
    ]
    return RunData(
        run_id, village, seed, status, days, load_models(conn, run_id), load_fish(conn, run_id),
        dict(catches), guesses, messages,
    )


def collect_runs(runs_dir: str | Path, experiment: str = "exp1") -> list[RunData]:
    """runs 폴더에서 이 실험의 끝난 마을만. 같은 마을·시드는 가장 최근(started_at) 것 하나."""
    prefix = f"{experiment}-"
    best: dict[tuple[str, int], tuple[str, Path, str, str]] = {}
    for path in sorted(Path(runs_dir).glob("*.db")):
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT run_id, condition, seed, status, started_at FROM runs").fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            continue
        for run_id, cond, seed, status, started in rows:
            if not cond.startswith(prefix) or status not in FINISHED:
                continue
            key = (cond[len(prefix):], seed)
            if key not in best or started > best[key][3]:
                best[key] = (run_id, path, status, started)
    out: list[RunData] = []
    for (village, seed), (run_id, path, status, _t) in sorted(best.items()):
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            out.append(_read_run(conn, run_id, village, seed, status))
        finally:
            conn.close()
    return out


# ---------------------------------------------------------------- 계산


def gini(values: list[float]) -> float:
    """지니 계수. 0 = 모두 같음, 1에 가까울수록 한 사람이 다 가짐. 모두 0이면 0."""
    xs = sorted(values)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


@dataclass
class RunMetrics:
    village: str
    group: str
    seed: int
    days: int
    collapsed: bool
    avg_stock: float
    efficiency: float
    gini: float
    day1_request: int  # 1일째 마을 전체 요청 (P6에서 본 핵심 숫자)
    total_catch: int


def run_metrics(r: RunData) -> RunMetrics:
    n_days = len(r.days)
    total = sum(d["total_catch"] for d in r.days)
    day1 = sum(f["amount"] for f in r.fish if f["day"] == 1)
    return RunMetrics(
        village=r.village, group=r.group, seed=r.seed, days=n_days,
        collapsed=any(d["collapsed"] for d in r.days),
        avg_stock=sum(d["stock_end"] for d in r.days) / n_days if n_days else 0.0,
        efficiency=total / (SUSTAINABLE_TOTAL * n_days) if n_days else 0.0,
        gini=gini(list(r.catches.values()) or [0]),
        day1_request=day1, total_catch=total,
    )


def role(r: RunData, agent: str) -> str:
    """이 어부가 마을에서 어떤 자리인지: S(같은 모델끼리), X(섞인 마을), O다수 / O한명, solo, off."""
    g = r.group
    if g != "O":
        return g
    mine = r.models.get(agent)
    same = sum(1 for m in r.models.values() if m == mine)
    return "O-한명" if same == 1 else "O-다수"


@dataclass
class ModelStats:
    """(모델, 자리)별로 모은 숫자."""

    agent_days: int = 0
    request_sum: int = 0
    inspects: int = 0
    parse_errors: int = 0
    says: int = 0
    num_says: int = 0  # "N마리"가 들어간 말
    kept: int = 0  # 그중 그날 요청 ≤ 말 속 가장 큰 N
    guesses_first: list[int] = field(default_factory=list)  # 첫 일기의 sustainable_guess
    guesses_last: list[int] = field(default_factory=list)  # 마지막 일기

    @property
    def avg_request(self) -> float:
        return self.request_sum / self.agent_days if self.agent_days else 0.0


def model_stats(runs: list[RunData]) -> dict[tuple[str, str], ModelStats]:
    out: dict[tuple[str, str], ModelStats] = defaultdict(ModelStats)
    for r in runs:
        req = {(f["day"], f["agent"]): f["amount"] for f in r.fish}
        for f in r.fish:
            s = out[(r.models.get(f["agent"], "?"), role(r, f["agent"]))]
            s.agent_days += 1
            s.request_sum += f["amount"]
            s.inspects += int(f["inspected"])
            s.parse_errors += int(f["parse_error"])
        for m in r.messages:
            s = out[(r.models.get(m["from"], "?"), role(r, m["from"]))]
            s.says += 1
            nums = [int(n) for n in NUM_FISH.findall(m["text"])]
            if nums:
                s.num_says += 1
                if req.get((m["day"], m["from"]), 0) <= max(nums):
                    s.kept += 1
        by_agent: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for day, agent, g in r.guesses:
            by_agent[agent].append((day, g))
        for agent, gs in by_agent.items():
            gs.sort()
            s = out[(r.models.get(agent, "?"), role(r, agent))]
            s.guesses_first.append(gs[0][1])
            s.guesses_last.append(gs[-1][1])
    return dict(out)


def mixing_effect(stats: dict[tuple[str, str], ModelStats]) -> list[dict]:
    """모델마다: 같은 모델 마을(S) 평균 요청과, 섞인 자리(X·O다수·O한명)에서의 평균 요청 차이."""
    rows = []
    for model in sorted({m for m, _ in stats}):
        base = stats.get((model, "S"))
        if base is None or not base.agent_days:
            continue
        for place in ("X", "O-다수", "O-한명"):
            s = stats.get((model, place))
            if s is None or not s.agent_days:
                continue
            rows.append({"model": model, "place": place, "S": base.avg_request,
                         "mixed": s.avg_request, "diff": s.avg_request - base.avg_request})
    return rows


def stock_series(r: RunData, length: int) -> list[float | None]:
    """날짜별 저녁 재고. 붕괴한 뒤 날은 0 (연못은 되살아나지 않는다). 기록이 없는 날은 None."""
    by_day = {d["day"]: d["stock_end"] for d in r.days}
    collapsed_on = next((d["day"] for d in r.days if d["collapsed"]), None)
    out: list[float | None] = []
    for day in range(1, length + 1):
        if day in by_day:
            out.append(by_day[day])
        elif collapsed_on is not None and day > collapsed_on:
            out.append(0)
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------- 표 (마크다운)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _end(m: RunMetrics) -> str:
    return f"{m.days}일째 붕괴" if m.collapsed else f"{m.days}일 버팀"


VILLAGE_ORDER = ("S", "X", "O", "solo", "off")


def _village_sort(v: str) -> tuple[int, str]:
    g = v.split("-", 1)[0]
    return (VILLAGE_ORDER.index(g) if g in VILLAGE_ORDER else 99, v)


def village_table(metrics: list[RunMetrics]) -> list[list[str]]:
    rows = [["마을", "시드별 끝", "버틴 마을", "평균 진행일", "1일째 총요청", "평균 잔존량", "효율", "불평등(지니)"]]
    for v in sorted({m.village for m in metrics}, key=_village_sort):
        ms = sorted((m for m in metrics if m.village == v), key=lambda m: m.seed)
        rows.append([
            v, " / ".join(_end(m) for m in ms), f"{sum(not m.collapsed for m in ms)}/{len(ms)}",
            f"{_mean([m.days for m in ms]):.1f}", " · ".join(str(m.day1_request) for m in ms),
            f"{_mean([m.avg_stock for m in ms]):.1f}", f"{_mean([m.efficiency for m in ms]):.2f}",
            f"{_mean([m.gini for m in ms]):.2f}",
        ])
    return rows


PLACE_LABELS = {"S": "같은 모델끼리", "X": "섞인 마을", "O-다수": "한 명만 다른 마을의 다수",
                "O-한명": "한 명만 다른 마을의 그 한 명", "solo": "혼자 말하기", "off": "대화 끔"}
PLACE_ORDER = ("S", "X", "O-다수", "O-한명", "solo", "off")


def model_table(stats: dict[tuple[str, str], ModelStats]) -> list[list[str]]:
    rows = [["모델", "자리", "어부-일", "평균 요청", "확인/어부-일", "말 수", "'N마리' 말", "그 말 지킴",
             "괜찮은 양 짐작 (첫→끝)", "해석 실패"]]
    keys = sorted(stats, key=lambda k: (k[0], PLACE_ORDER.index(k[1]) if k[1] in PLACE_ORDER else 99))
    for model, place in keys:
        s = stats[(model, place)]
        kept = f"{s.kept}/{s.num_says}" if s.num_says else "-"
        guess = (f"{_mean(s.guesses_first):.1f} → {_mean(s.guesses_last):.1f}" if s.guesses_first else "-")
        rows.append([
            model, PLACE_LABELS.get(place, place), str(s.agent_days), f"{s.avg_request:.1f}",
            f"{s.inspects / s.agent_days:.2f}" if s.agent_days else "-", str(s.says), str(s.num_says), kept,
            guess, str(s.parse_errors),
        ])
    return rows


def mixing_table(rows: list[dict]) -> list[list[str]]:
    out = [["모델", "섞인 자리", "같은 모델 마을 평균 요청", "섞인 자리 평균 요청", "차이"]]
    for r in rows:
        out.append([r["model"], PLACE_LABELS[r["place"]], f"{r['S']:.1f}", f"{r['mixed']:.1f}", f"{r['diff']:+.1f}"])
    return out


def markdown(rows: list[list[str]]) -> str:
    head, *body = rows
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ---------------------------------------------------------------- 그래프 (SVG, 라이브러리 없음)

MODEL_SLOTS = ("var(--s1)", "var(--s2)", "var(--s3)")  # 모델 색은 모델 이름 순서로 고정 (순위 아님)


def _esc(x: object) -> str:
    return html.escape(str(x))


def svg_stock_panels(runs: list[RunData], length: int) -> str:
    """마을 종류마다 작은 그래프 하나: 시드별 저녁 재고(가는 선) + 평균(굵은 선). 같은 축 0~100."""
    villages = sorted({r.village for r in runs}, key=_village_sort)
    w, h, pl, pr, pt, pb = 260, 170, 34, 10, 24, 26
    iw, ih = w - pl - pr, h - pt - pb
    panels = []
    for v in villages:
        rs = sorted((r for r in runs if r.village == v), key=lambda r: r.seed)
        x = lambda d: pl + (d - 1) / max(length - 1, 1) * iw  # noqa: E731
        y = lambda s: pt + ih - s / 100 * ih  # noqa: E731
        parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(v)} 날짜별 저녁 재고">',
                 f'<text x="{pl}" y="15" class="pt">{_esc(v)}</text>']
        for s in (0, 50, 100):
            parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(s):.1f}" y2="{y(s):.1f}" class="grid"/>'
                         f'<text x="{pl - 6}" y="{y(s) + 4:.1f}" class="ax" text-anchor="end">{s}</text>')
        for d in (1, length):
            parts.append(f'<text x="{x(d):.1f}" y="{h - 8}" class="ax" text-anchor="middle">{d}일</text>')
        series = [stock_series(r, length) for r in rs]
        for r, ser in zip(rs, series):
            pts = [(x(i + 1), y(s)) for i, s in enumerate(ser) if s is not None]
            if pts:
                parts.append('<polyline class="seed" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
                             + f'"><title>시드 {r.seed}</title></polyline>')
        mean_pts = []
        for i in range(length):
            vals = [s[i] for s in series if s[i] is not None]
            if vals:
                mean_pts.append((i + 1, _mean(vals)))
        if mean_pts:
            parts.append('<polyline class="mean" points="'
                         + " ".join(f"{x(d):.1f},{y(s):.1f}" for d, s in mean_pts) + '"/>')
            for d, s in mean_pts:
                parts.append(f'<circle class="hit" cx="{x(d):.1f}" cy="{y(s):.1f}" r="7">'
                             f'<title>{_esc(v)} {d}일째 평균 저녁 재고 {s:.1f}</title></circle>')
        parts.append("</svg>")
        panels.append("".join(parts))
    return '<div class="panels">' + "".join(f'<figure class="panel">{p}</figure>' for p in panels) + "</div>"


def svg_request_bars(stats: dict[tuple[str, str], ModelStats]) -> str:
    """자리(같은 모델 / 섞인 / 다수 / 한 명)마다 모델별 평균 요청 막대. 점선 = 1인 10마리."""
    models = sorted({m for m, _ in stats})
    places = [p for p in ("S", "X", "O-다수", "O-한명") if any((m, p) in stats for m in models)]
    if not models or not places:
        return "<p class='muted'>기록이 없다.</p>"
    vmax = max(20.0, *(s.avg_request for s in stats.values()))
    bw, gap, ggap = 26, 2, 34
    pl, pt, pb = 34, 16, 46
    gw = len(models) * bw + (len(models) - 1) * gap
    w = pl + len(places) * gw + (len(places) - 1) * ggap + 20
    h, ih = 240, 240 - pt - pb
    y = lambda v: pt + ih - v / vmax * ih  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="모델별 평균 요청">']
    for t in (0, 5, 10, 15, 20):
        cls = "ref" if t == SUSTAINABLE_EACH else "grid"
        parts.append(f'<line x1="{pl}" x2="{w - 10}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{cls}"/>'
                     f'<text x="{pl - 6}" y="{y(t) + 4:.1f}" class="ax" text-anchor="end">{t}</text>')
    for gi, p in enumerate(places):
        gx = pl + 10 + gi * (gw + ggap)
        for mi, m in enumerate(models):
            s = stats.get((m, p))
            if s is None or not s.agent_days:
                continue
            bx, top = gx + mi * (bw + gap), min(y(s.avg_request), y(0) - 4)
            parts.append(
                f'<path d="M{bx},{y(0):.1f} V{top + 4:.1f} Q{bx},{top:.1f} {bx + 4},{top:.1f} H{bx + bw - 4}'
                f' Q{bx + bw},{top:.1f} {bx + bw},{top + 4:.1f} V{y(0):.1f} Z" style="fill:{MODEL_SLOTS[mi % 3]}"'
                f' class="bar"><title>{_esc(m)} · {_esc(PLACE_LABELS[p])}: 평균 {s.avg_request:.1f}마리'
                f' (어부-일 {s.agent_days})</title></path>'
                f'<text x="{bx + bw / 2}" y="{top - 4:.1f}" class="val" text-anchor="middle">{s.avg_request:.1f}</text>'
            )
        label = PLACE_LABELS[p].replace("한 명만 다른 마을의 ", "")
        parts.append(f'<text x="{gx + gw / 2}" y="{h - 26}" class="ax" text-anchor="middle">{_esc(label)}</text>')
    parts.append(f'<text x="{w - 12}" y="{y(SUSTAINABLE_EACH) - 5:.1f}" class="ax" text-anchor="end">1인 10마리</text>')
    parts.append("</svg>")
    legend = "".join(f'<span><i style="background:{MODEL_SLOTS[i % 3]}"></i>{_esc(m)}</span>'
                     for i, m in enumerate(models))
    return f'<div class="legend">{legend}</div><div class="scroll">{"".join(parts)}</div>'


def svg_day1(metrics: list[RunMetrics]) -> str:
    """마을 종류별 1일째 마을 전체 요청: 시드별 점 + 평균 막대. 점선 = 50 (넘으면 연못이 반 아래로)."""
    villages = sorted({m.village for m in metrics}, key=_village_sort)
    if not villages:
        return ""
    vmax = max(100, *(m.day1_request for m in metrics))
    pl, pt, row = 118, 10, 30
    w, h = 560, pt + row * len(villages) + 30
    iw = w - pl - 20
    x = lambda v: pl + v / vmax * iw  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="1일째 마을 전체 요청">']
    for t in range(0, int(vmax) + 1, 25):
        cls = "ref" if t == SUSTAINABLE_TOTAL else "grid"
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{pt}" y2="{h - 24}" class="{cls}"/>'
                     f'<text x="{x(t):.1f}" y="{h - 8}" class="ax" text-anchor="middle">{t}</text>')
    for i, v in enumerate(villages):
        cy = pt + i * row + row / 2
        ms = [m for m in metrics if m.village == v]
        avg = _mean([m.day1_request for m in ms])
        parts.append(f'<text x="{pl - 8}" y="{cy + 4:.1f}" class="ax" text-anchor="end">{_esc(v)}</text>'
                     f'<rect x="{pl}" y="{cy - 6:.1f}" width="{max(x(avg) - pl, 0.5):.1f}" height="12" rx="4"'
                     f' class="avgbar"><title>{_esc(v)} 평균 {avg:.1f}</title></rect>')
        for m in ms:
            parts.append(f'<circle cx="{x(m.day1_request):.1f}" cy="{cy:.1f}" r="5" class="dot">'
                         f'<title>{_esc(v)} 시드 {m.seed}: {m.day1_request}마리 · {_esc(_end(m))}</title></circle>')
    parts.append("</svg>")
    return f'<div class="scroll">{"".join(parts)}</div>'


def html_table(rows: list[list[str]]) -> str:
    head, *body = rows
    th = "".join(f"<th>{_esc(c)}</th>" for c in head)
    trs = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


CSS = """
:root{--bg:#f2f7f6;--card:#ffffff;--fg:#1f2b2a;--muted:#5d6e6c;--line:#d3e1de;--grid:#e3ecea;--ref:#c0392b;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--seed:#9fb3b0;--mean:#1f6f7a;--dot:#1f6f7a;--avg:#cfe5e8;
  --cute:"Jua","Noto Sans KR",system-ui,sans-serif;--sans:"Noto Sans KR",system-ui,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141d1c;--card:#1c2928;--fg:#e6efed;
  --muted:#9db1ae;--line:#2e4240;--grid:#263735;--ref:#ff8a7a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
  --seed:#55706c;--mean:#6fd0db;--dot:#6fd0db;--avg:#24474b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141d1c;--card:#1c2928;--fg:#e6efed;--muted:#9db1ae;--line:#2e4240;--grid:#263735;
  --ref:#ff8a7a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--seed:#55706c;--mean:#6fd0db;--dot:#6fd0db;--avg:#24474b;
  color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.6;font-size:14px;
  padding:24px max(16px,env(safe-area-inset-right,0px)) 40px max(16px,env(safe-area-inset-left,0px))}
main{max-width:1000px;margin:0 auto;display:grid;gap:22px}
h1,h2{font-family:var(--cute);font-weight:400;margin:0}h1{font-size:2rem}h2{font-size:1.3rem}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;display:grid;gap:10px;min-width:0}
.muted{color:var(--muted)}p{margin:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:.8rem}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--muted);font-family:var(--sans);font-size:11px}
svg .pt{fill:var(--fg);font-size:13px;font-weight:700}
svg .val{fill:var(--fg);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}.ref{stroke:var(--ref);stroke-width:1.5;stroke-dasharray:4 4}
.seed{fill:none;stroke:var(--seed);stroke-width:1.5}.mean{fill:none;stroke:var(--mean);stroke-width:2.5}
.hit{fill:transparent}.hit:hover{fill:var(--mean)}
.dot{fill:var(--dot);stroke:var(--card);stroke-width:2}.avgbar{fill:var(--avg)}
.bar:hover{opacity:.8}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.panel{margin:0;min-width:0}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
"""


def render_report(runs: list[RunData], experiment: str) -> str:
    metrics = [run_metrics(r) for r in runs]
    stats = model_stats(runs)
    mix = mixing_effect(stats)
    length = max((len(r.days) for r in runs), default=10)
    length = max(length, max((m.days for m in metrics), default=1))
    n = len(runs)
    villages = len({r.village for r in runs})
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>연못마을 {experiment} 결과</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Jua&family=Noto+Sans+KR:wght@400;500;700&display=swap">
<style>{CSS}</style></head><body><main>
<header><h1>🎣 {experiment} 모델 마을 결과</h1>
<p class="muted">마을 {villages}종 · 실행 {n}번. 숫자는 전부 runs/*.db 기록에서 계산했다. 마우스를 올리면 값이 보인다.</p></header>
<section><h2>날짜별 저녁 재고</h2>
<p class="muted">가는 선 = 시드 하나, 굵은 선 = 평균. 붕괴한 뒤는 0으로 그렸다 (연못은 되살아나지 않는다).</p>
{svg_stock_panels(runs, length)}</section>
<section><h2>첫날 마을 전체 요청</h2>
<p class="muted">점 = 시드 하나, 막대 = 평균. 빨간 점선 50을 넘기면 연못이 반 아래로 내려간다.</p>
{svg_day1(metrics)}</section>
<section><h2>모델별 평균 요청 (어부 1명 하루)</h2>
<p class="muted">같은 모델이 자기들끼리일 때와 섞였을 때. 빨간 점선 = 1인 10마리 (5명이 나누면 연못이 유지되는 양).</p>
{svg_request_bars(stats)}</section>
<section><h2>마을별 지표</h2>{html_table(village_table(metrics))}</section>
<section><h2>모델별 지표</h2>{html_table(model_table(stats))}
<p class="muted">'N마리' 말: 말 속에 "N마리"가 들어간 말. 지킴: 그날 내 요청이 말 속 가장 큰 N 이하.
"어제 15마리를 잡았다"처럼 약속이 아닌 말도 섞이는 어림 지표다.</p></section>
<section><h2>섞임 효과</h2>{html_table(mixing_table(mix)) if mix else '<p class="muted">비교할 기록이 없다.</p>'}
<p class="muted">차이 = 섞인 자리 평균 요청 − 같은 모델 마을 평균 요청. 시드 3개라 작은 차이는 우연일 수 있다.</p></section>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실험 마을 기록으로 지표 표와 그래프를 만든다 (읽기만, 비용 0원).")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--experiment", default="exp1")
    parser.add_argument("--out", default=None, help="보고서 HTML (기본 <runs-dir>/<실험>_report.html)")
    args = parser.parse_args(argv)

    runs = collect_runs(args.runs_dir, args.experiment)
    if not runs:
        print(f"{args.experiment} 기록이 없습니다. 먼저: uv run python -m pondvillage.experiment")
        return 1
    metrics = [run_metrics(r) for r in runs]
    stats = model_stats(runs)
    print(f"## 마을별 ({len(runs)}번)\n")
    print(markdown(village_table(metrics)))
    print("\n## 모델별\n")
    print(markdown(model_table(stats)))
    mix = mixing_effect(stats)
    if mix:
        print("\n## 섞임 효과 (평균 요청)\n")
        print(markdown(mixing_table(mix)))
    out = Path(args.out) if args.out else Path(args.runs_dir) / f"{args.experiment}_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(runs, args.experiment), encoding="utf-8")
    print(f"\n보고서: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
