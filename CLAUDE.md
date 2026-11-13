# CLAUDE.md — 연못마을 (Pond Village)

LLM 에이전트 5명이 공동 연못 하나를 나눠 쓰는 어촌 마을 시뮬레이션.
1단계: 공유 자원을 지키는가(공유지의 비극) → 2단계: 거짓 소문이 자원 행동을 바꾸는가.

작업 전에 반드시 읽을 문서:
- `docs/SPEC.md` — 월드·규칙·데이터 구조 (규칙의 정답지)
- `docs/EXPERIMENT.md` — 실험 질문·조건·지표
- `docs/IMP-PLAN.md` — 단계별 구현 계획과 완료 기준
- `TODO.md` — 지금 할 일

## 기술 스택
- Python 3.11+, 패키지 관리는 `uv`
- LLM: OpenAI 호환 API 하나로 통일 (`openai` 패키지)
  - 개발: Ollama 로컬 (`http://localhost:11434/v1`)
  - 본 실험: 저가 유료 모델 (OpenRouter 등) — `base_url`과 `model`만 바꿈
- 저장: SQLite (`runs/*.db`)
- 설정: YAML (`configs/`)
- 대시보드: Streamlit
- 테스트: pytest

## 폴더 구조
```
pond-village/
  CLAUDE.md  TODO.md
  docs/        SPEC.md  EXPERIMENT.md  IMP-PLAN.md  NOTES.md(설계 결정 기록)
  configs/     base.yaml  exp1_*.yaml  exp2_*.yaml
  src/pondvillage/
    world.py      # 월드 상태 (연못, 에이전트, 날짜)
    rules.py      # 규칙 엔진: 어획 배분, 재생, 정산, 사망 판정 (LLM 없음)
    actions.py    # 행동 정의(도구 스키마)와 검증
    agents/
      base.py     # 에이전트 인터페이스
      scripted.py # 규칙 기반 봇 (탐욕/협력/랜덤) — 검증·기준선용
      llm_agent.py
    memory.py     # 기억 저장·요약
    llm.py        # LLM 호출은 전부 여기서만
    rumor.py      # (2단계) 소문 주입
    runner.py     # 하루 단계 진행, 실험 실행
    logger.py     # SQLite 기록
  dashboard/app.py
  tests/
  runs/          # 실행 결과 (git 제외)
```

## 반드시 지킬 규칙
1. **판정은 코드가 한다.** 어획량 배분, 연못 재생, 돈, 사망은 전부 `rules.py`에서 결정한다. LLM은 "무엇을 하고 싶은지"만 말한다.
2. **LLM 호출은 `llm.py`에서만 한다.** 모든 호출의 프롬프트·응답·토큰 수·지연시간·추정 비용을 `llm_calls` 테이블에 남긴다.
3. **유료 API는 명시적으로 켰을 때만 쓴다.** 기본값은 Ollama. 유료 호출은 `--paid` 플래그와 설정의 `cost_cap_usd`가 둘 다 있어야 실행되고, 상한을 넘으면 즉시 멈춘다.
4. **모든 무작위성은 시드로 고정한다.** `random.Random(seed)` 인스턴스를 넘겨 쓰고, 전역 random은 쓰지 않는다.
5. **SPEC의 규칙·숫자를 임의로 바꾸지 않는다.** 바꿔야 할 것 같으면 먼저 제안하고, 승인되면 SPEC과 `docs/NOTES.md`에 이유를 기록한 뒤 바꾼다.
6. **LLM 응답 파싱 실패에 대비한다.** JSON 파싱 실패 시 1회 재시도, 그래도 실패하면 "아무것도 안 함" 행동으로 처리하고 로그에 `parse_error`로 남긴다. 절대 크래시하지 않는다.
7. **규칙 엔진에는 테스트가 있어야 한다.** `rules.py`를 바꾸면 `tests/test_rules.py`를 같이 고친다.

## 작업 방식
- **한 번에 TODO 한 항목만** 구현한다. 끝나면 TODO에 체크하고 멈춘다.
- 작업이 끝나면 아래 세 가지를 짧게 알려준다.
  1. 무엇을 바꿨는지 (파일 단위, 3줄 이내)
  2. 핵심 설계 결정과 이유 (내가 면접에서 설명할 수 있게)
  3. 내가 직접 확인할 명령어 (예: `uv run pytest`, `uv run python -m pondvillage.runner --config configs/base.yaml --days 2`)
- 새 라이브러리를 추가하기 전에 먼저 물어본다.
- 범위 밖의 개선(리팩터링, 기능 추가)은 하지 말고 `TODO.md`의 "나중에" 칸에 제안만 적는다.
