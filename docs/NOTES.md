# NOTES — 설계 결정 기록

> SPEC의 규칙·숫자를 바꾸거나 중요한 설계를 정할 때, 날짜·결정·이유를 남긴다.

## 2026-11-14 · 프로젝트 셋업
- `uv init --package` (src 레이아웃) 사용. 이유: 테스트가 설치된 패키지 `pondvillage`를 import하게 되어, 작업 폴더 경로에 우연히 기대는 실수를 막는다.
- pytest는 개발 의존성(`dependency-groups.dev`)으로만 추가. 실행 코드 의존성과 분리.

## 2026-11-14 · 로컬 모델 (Ollama)
- 새로 설치하지 않고 이 PC에 이미 있는 Ollama를 쓴다. Windows가 아니라 Ubuntu(WSL) 안에서 돌고, 주소는 그대로 `http://localhost:11434/v1`. Ubuntu가 쉬면 Ollama도 같이 꺼지므로, 응답이 없으면 Ubuntu를 먼저 켠다.
- 받아져 있는 모델: `qwen2.5vl:7b`(8.3B, 약 6GB), `gemma3:4b`(3.9B).
- 어획 JSON을 한 번씩 시켜 본 결과(각 약 5초): `qwen2.5vl:7b`는 JSON만 답함. `gemma3:4b`는 JSON을 ```` ```json ```` 코드 블록으로 감싸서 답함 → 파서가 코드 블록 껍질을 벗겨야 한다(P4에서 반영).
- 개발용 모델을 무엇으로 정할지는 TODO "결정 필요"에 남겨 둔다. 한 번씩만 해 본 것이라 아직 결론이 아니다.

## 2026-11-14 · 작업 지침 파일 구조
- 컷토리 프로젝트와 같은 방식으로 나눔: 공통 지침은 `AGENTS.md`(Codex도 읽음), `CLAUDE.md`는 `@AGENTS.md` 한 줄로 불러오고 Claude 전용 규칙만 둔다. 이유: 지침을 한 곳에서만 고치면 되고, 도구를 바꿔도 같은 규칙이 적용된다.

## 2026-11-14 · llm.py 최소 버전
- `openai` 패키지 하나로 호출한다. Ollama도 유료 서비스도 같은 형식(OpenAI 호환 API)을 따르므로 `base_url`과 `model`만 바꾸면 된다.
- 로컬 주소(localhost, 127.0.0.1)가 아니면 `PaidCallNotAllowed`로 호출 전에 멈춘다. `--paid`와 `cost_cap_usd`가 생기는 P4 전에 실수로 돈이 나가는 것을 막기 위해서다.
- `chat()`은 답과 함께 토큰 수·걸린 시간을 `LLMResult`로 돌려준다. `llm_calls` 테이블에 넣을 값을 미리 모아 둔 것이고, 실제 기록은 로그가 생기는 P3~P4에서 붙인다.
- 기본 모델 `qwen2.5vl:7b`는 임시값이다. 개발용 모델을 정하면 `configs/base.yaml`에서 읽도록 바꾼다.
- 테스트는 가짜 클라이언트를 넣어 네트워크 없이 돌린다(`client` 인자를 바깥에서 넣을 수 있게 한 이유).
