"""실험 러너: `configs/exp1_*.yaml`에 적은 마을들을 시드마다 차례로 돌린다 (EXPERIMENT 1단계, P8).

실험 파일 하나 = 마을 여러 개 × 시드 여러 개. 마을마다 어부 자리 순서대로 모델을 적는다.
- rotate: true면 시드마다 모델 자리를 한 칸씩 돌린다. 섞인 마을에서 "만석은 늘 qwen" 같은
  이름 탓 차이가 생기지 않게 하려고 (EXPERIMENT 1단계 조건 X).
- talk: 마을마다 base 설정의 talk 칸을 덮어쓸 수 있다 (혼자 말하기·대화 끈 마을).
- 이미 끝난 마을(같은 condition·시드, 끝난 이유가 적힌 기록)은 건너뛴다. 끊겨도 같은 명령으로 이어 돌린다.

기록은 마을 하나 = `runs/<시각>_<실험>-<마을>_s<시드>.db` 하나. runs.condition = "<실험>-<마을>".
판정·LLM 호출은 전부 러너(`runner.run_llm_village`)가 한다. 여기서는 목록만 만든다.

실행 예:
    uv run python -m pondvillage.experiment --dry-run          # 돌릴 목록과 자리만 보기 (호출 0번)
    uv run python -m pondvillage.experiment                    # configs/exp1_*.yaml 전부
    uv run python -m pondvillage.experiment configs/exp1_same.yaml
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from pondvillage import llm
from pondvillage.logger import make_run_id, make_stamp
from pondvillage.runner import END_LABELS, print_progress, run_llm_village, talk_label, village_label
from pondvillage.world import Config, ConfigError, config_from_dict

FINISHED = set(END_LABELS)  # 끝난 이유가 적힌 기록만 "끝난 마을"로 본다 (running·error·aborted_cost는 다시 돌림)


@dataclass(frozen=True)
class Village:
    """실험 파일에 적은 마을 하나."""

    name: str  # 예: S-qwen. 앞 글자(- 앞)가 조건 묶음 (S·X·O·solo·off)
    models: tuple[str, ...]  # 어부 자리 순서대로
    rotate: bool = False
    talk: dict | None = None  # base의 talk 칸을 덮어쓸 값


@dataclass(frozen=True)
class Experiment:
    name: str  # 예: exp1
    config_name: str  # 실험 파일 이름 (runs.config_name)
    base_raw: dict  # base 설정 파일을 읽은 dict
    days: int
    seeds: tuple[int, ...]
    villages: tuple[Village, ...]


@dataclass(frozen=True)
class PlannedRun:
    """실제로 돌릴 한 번: 마을 × 시드."""

    experiment: str
    config_name: str
    village: Village
    seed: int
    days: int
    config: Config
    models: dict[str, str]  # 어부 id → 모델 (돌린 뒤)

    @property
    def condition(self) -> str:
        return f"{self.experiment}-{self.village.name}"


def rotate_models(models: tuple[str, ...], seed: int) -> list[str]:
    """시드 s면 오른쪽으로 s-1칸 돌린다. 시드 1은 그대로. 예: qgqgq → (시드 2) qqgqg → (시드 3) gqqgq."""
    n = len(models)
    shift = (seed - 1) % n
    return [models[(i - shift) % n] for i in range(n)]


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        base_path = path.parent / raw["base"]
        with open(base_path, encoding="utf-8") as f:
            base_raw = yaml.safe_load(f)
        villages = tuple(
            Village(name=v["name"], models=tuple(v["models"]), rotate=bool(v.get("rotate", False)), talk=v.get("talk"))
            for v in raw["villages"]
        )
        exp = Experiment(
            name=raw["experiment"], config_name=path.stem, base_raw=base_raw,
            days=int(raw["days"]), seeds=tuple(int(s) for s in raw["seeds"]), villages=villages,
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"실험 파일 {path} 항목이 빠졌거나 이름이 틀렸습니다: {e}") from e
    n_agents = len(base_raw["agents"])
    for v in villages:
        if len(v.models) != n_agents:
            raise ConfigError(f"{v.name}: 모델 {len(v.models)}개, 어부는 {n_agents}명입니다")
    return exp


def village_config(base_raw: dict, village: Village) -> Config:
    """base 설정에 마을의 talk 덮어쓰기를 얹는다. 숫자 규칙(연못·경제)은 건드리지 않는다."""
    raw = dict(base_raw)
    if village.talk:
        raw["talk"] = {**(base_raw.get("talk") or {}), **village.talk}
    return config_from_dict(raw)


def plan(experiments: list[Experiment]) -> list[PlannedRun]:
    out: list[PlannedRun] = []
    for exp in experiments:
        for village in exp.villages:
            config = village_config(exp.base_raw, village)
            for seed in exp.seeds:
                seats = rotate_models(village.models, seed) if village.rotate else list(village.models)
                models = {spec.id: seats[i] for i, spec in enumerate(config.agents)}
                out.append(PlannedRun(exp.name, exp.config_name, village, seed, exp.days, config, models))
    return out


def finished_runs(runs_dir: Path) -> set[tuple[str, int]]:
    """runs 폴더에서 이미 끝난 (condition, 시드). 깨진 파일은 건너뛴다."""
    done: set[tuple[str, int]] = set()
    for db in sorted(runs_dir.glob("*.db")):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT condition, seed, status FROM runs").fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            continue
        done.update((c, s) for c, s, st in rows if st in FINISHED)
    return done


def seating(config: Config, models: dict[str, str]) -> str:
    """자리 표시. 예: 만석 qwen2.5:7b · 순자 gemma3:4b ..."""
    return " · ".join(f"{spec.name} {models[spec.id]}" for spec in config.agents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실험 파일의 마을들을 시드마다 차례로 돌린다 (로컬 Ollama).")
    parser.add_argument("files", nargs="*", help="실험 파일 (기본: configs/exp1_*.yaml 전부)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true", help="돌릴 목록과 자리만 보여 주고 끝낸다")
    args = parser.parse_args(argv)

    files = args.files or sorted(str(p) for p in Path("configs").glob("exp1_*.yaml"))
    if not files:
        print("실험 파일이 없습니다 (configs/exp1_*.yaml).")
        return 1
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    planned = plan([load_experiment(f) for f in files])
    done = finished_runs(runs_dir)
    todo = [p for p in planned if (p.condition, p.seed) not in done]

    print(f"실험 파일 {len(files)}개 · 마을 × 시드 {len(planned)}번 · 이미 끝남 {len(planned) - len(todo)}번 · 남음 {len(todo)}번")
    for i, p in enumerate(planned, 1):
        mark = "끝남" if (p.condition, p.seed) in done else "남음"
        print(f"  [{mark}] {p.condition} 시드 {p.seed} · {p.days}일 · 대화 {talk_label(p.config)} · {seating(p.config, p.models)}")
    if args.dry_run or not todo:
        return 0

    # 출발 전 점검 한 번 (꺼진 서버에 마을마다 기다리지 않게). 유료 주소는 여기서 거절된다.
    base_url = todo[0].config.llm.base_url if todo[0].config.llm else llm.DEFAULT_BASE_URL
    all_models = sorted({m for p in todo for m in p.models.values()})
    try:
        llm.check_ollama(base_url, all_models)
    except (llm.LLMUnavailable, llm.PaidCallNotAllowed) as e:
        print(f"[시작 못 함] {e}")
        return 1

    started = time.monotonic()
    for i, p in enumerate(todo, 1):
        print(f"\n▶ ({i}/{len(todo)}) {p.condition} 시드 {p.seed} · {village_label(p.models)} · 대화 {talk_label(p.config)}")
        t0 = time.monotonic()
        vr = run_llm_village(
            p.config, p.models, p.seed, run_id=make_run_id(make_stamp(), p.condition, p.seed),
            condition=p.condition, config_name=p.config_name, runs_dir=runs_dir, days=p.days, on_day=print_progress,
        )
        mins = (time.monotonic() - t0) / 60
        if vr.result is None:
            # 로컬에서 멈추는 까닭은 거의 늘 서버가 꺼진 것이라, 다음 마을도 실패한다. 실험 전체를 멈춘다.
            print(f"  [멈춤] {vr.error}")
            print("  같은 명령으로 다시 돌리면 끝난 마을은 건너뛰고 이 마을부터 다시 한다.")
            return 1
        bad = sum(1 for c in vr.calls if c.parse_error)
        print(f"  끝: {END_LABELS[vr.result.end_reason]} · {len(vr.result.days)}일 · 호출 {len(vr.calls)}번"
              f" (해석 실패 {bad}) · {mins:.1f}분 · {vr.db_path}")
    total = (time.monotonic() - started) / 60
    print(f"\n모두 끝: {len(todo)}번 완료 · {total:.0f}분")
    print("분석: uv run python -m pondvillage.analyze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
