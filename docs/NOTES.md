# NOTES — 설계 결정 기록

> SPEC의 규칙·숫자를 바꾸거나 중요한 설계를 정할 때, 날짜·결정·이유를 남긴다.

## 2026-11-14 · 프로젝트 셋업
- `uv init --package` (src 레이아웃) 사용. 이유: 테스트가 설치된 패키지 `pondvillage`를 import하게 되어, 작업 폴더 경로에 우연히 기대는 실수를 막는다.
- pytest는 개발 의존성(`dependency-groups.dev`)으로만 추가. 실행 코드 의존성과 분리.
