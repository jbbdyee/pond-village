"""개발용 로컬 모델 고르기: 같은 어부 시험지로 모델 여러 개를 비교한다.

실행: uv run python scripts/try_models.py qwen2.5:7b qwen2.5vl:7b gemma3:4b
결과: 콘솔 표 + runs/model_trial_<시각>.json (git 제외)

시험지는 P4 프롬프트의 맛보기다. 진짜 프롬프트 파일은 P4에서 prompts/에 만든다.
LLM 호출은 llm.chat만 쓴다(지침 규칙 2). 채점은 actions.py 검사를 그대로 쓴다.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from pondvillage import llm
from pondvillage.actions import FISH, INSPECT, catch_cap, validate_fish_action
from pondvillage.world import load_config

CONFIG = Path("configs/base.yaml")
REPEATS = 2

# (이름, 아침 추정치, 어제 총어획, 정확한 재고 또는 None)
SITUATIONS = [
    ("넉넉", 104, 50, None),
    ("줄어듦", 58, 70, None),
    ("확인후", 61, 70, 55),
]

SYSTEM = """너는 작은 어촌 마을의 어부다. 마을 사람 5명이 연못 하나를 나눠 쓴다.
규칙: 연못은 최대 100마리. 하루 끝에 남은 물고기가 2배로 늘어난다(최대 100). 남은 물고기가 5마리보다 적으면 연못이 영영 무너진다.
물고기 1마리는 1원에 팔리고, 생활비는 하루 5원이다. 돈이 0원 아래인 날이 3일 이어지면 마을을 떠난다.
한 사람은 하루 최대 20마리를 잡을 수 있다. 아침 공지의 재고는 추정치라 틀릴 수 있다.
"inspect_pond"로 정확한 재고를 확인할 수 있지만, 그날은 최대 10마리만 잡을 수 있다.
반드시 아래 JSON 하나로만 한국어로 답하라. 다른 글은 쓰지 마라.
{"action": "fish", "amount": 0~20 정수, "reason": "이유"}
또는 {"action": "inspect_pond", "reason": "이유"}"""


def user_prompt(name: str, persona: str, money: int, estimate: int, yesterday: int, exact: int | None) -> str:
    lines = [
        f"너의 이름은 {name}. {persona}",
        f"오늘 아침 공지: 연못 재고 추정치 약 {estimate}마리, 어제 마을 전체 어획 {yesterday}마리.",
        f"내 돈: {money}원.",
    ]
    if exact is not None:
        lines.append(f"너는 연못을 직접 확인했다. 정확한 재고는 {exact}마리다. 오늘은 최대 10마리까지 잡을 수 있다. 이제 fish로 답하라.")
    lines.append("오늘 무엇을 할지 JSON으로 답하라.")
    return "\n".join(lines)


def parse(text: str) -> object:
    """JSON 글자 → dict. 앞뒤 군말이나 ```json 상자가 있으면 첫 { ~ 마지막 }만 본다."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def trial(model: str, base_url: str, agents, economy) -> list[dict]:
    rows = []
    for json_mode in (True, False):
        for sit, estimate, yesterday, exact in SITUATIONS:
            for agent in agents:
                for rep in range(REPEATS):
                    messages = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt(agent.name, agent.persona, 20, estimate, yesterday, exact)},
                    ]
                    try:
                        res = llm.chat(messages, model=model, base_url=base_url, temperature=0.7, json_mode=json_mode)
                        text, latency = res.text, res.latency_ms
                    except Exception as error:  # 시험이므로 멈추지 않고 실패로 센다
                        text, latency = f"<호출 실패: {error}>", 0
                    data = parse(text)
                    dec = validate_fish_action(data, catch_cap(economy, exact is not None), allow_inspect=exact is None)
                    rows.append({
                        "model": model, "json_mode": json_mode, "situation": sit, "agent": agent.name,
                        "rep": rep, "text": text, "latency_ms": latency,
                        "json_ok": isinstance(data, dict), "valid": dec.parse_error is None,
                        "kind": dec.kind, "amount": dec.amount, "clamped": dec.clamped,
                        "korean_reason": korean(dec.reason),
                    })
                    print(f"  {model} json={json_mode} {sit} {agent.name}: {dec.kind} {dec.amount} ({latency}ms)", flush=True)
    return rows


def summarize(rows: list[dict]) -> None:
    print("\n모델 | JSON모드 | JSON됨 | 검사통과 | 한국어이유 | 평균ms | 넉넉일때 평균마릿수(만석/순자/미란)")
    keys = sorted({(r["model"], r["json_mode"]) for r in rows})
    for model, jm in keys:
        rs = [r for r in rows if r["model"] == model and r["json_mode"] == jm]
        n = len(rs)
        pct = lambda k: f"{sum(r[k] for r in rs)}/{n}"
        ms = round(sum(r["latency_ms"] for r in rs) / n)
        def avg(name):
            xs = [r["amount"] for r in rs if r["situation"] == "넉넉" and r["agent"] == name and r["kind"] == FISH]
            return f"{sum(xs)/len(xs):.1f}" if xs else "-"
        print(f"{model} | {jm} | {pct('json_ok')} | {pct('valid')} | {pct('korean_reason')} | {ms} | {avg('만석')}/{avg('순자')}/{avg('미란')}")


def main(argv: list[str]) -> int:
    models = argv or ["qwen2.5:7b", "qwen2.5vl:7b", "gemma3:4b"]
    config = load_config(CONFIG)
    base_url = llm.DEFAULT_BASE_URL
    rows = []
    for model in models:
        print(f"[{model}] 시험 시작")
        rows += trial(model, base_url, config.agents, config.economy)
    summarize(rows)
    out = Path("runs") / f"model_trial_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
