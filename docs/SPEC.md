# SPEC — 연못마을

> 이 문서가 규칙의 정답지다. 숫자는 초기값이고, 바꿀 땐 `docs/NOTES.md`에 이유를 남긴다.

## 1. 월드

### 연못
| 항목 | 값 | 설명 |
|---|---|---|
| `capacity` | 100 | 연못 최대 물고기 수 |
| `initial_stock` | 100 | 시작 물고기 수 |
| 재생 | `stock = min(capacity, stock * 2)` | 하루 끝에 남은 물고기가 2배로 (상한 100) |
| 붕괴 | `stock < 5` | 재생 전에 판정한다 (어획 후 남은 물고기 기준). 붕괴 시 재생 불가, 시뮬레이션 종료 |

- 지속 가능한 하루 총 어획량은 최대 **50마리**(100 → 50 → 재생 100). 5명이면 1인당 10마리.
- 이 숫자는 에이전트에게 직접 알려주지 않는다 (스스로 알아내는지가 관찰 포인트).

### 정보 비대칭 (2단계 소문 실험의 핵심)
- 에이전트는 정확한 `stock`을 모른다.
- 매일 아침 **추정치**만 공개된다: `round(stock * U(0.8, 1.2))` (시드 고정)
- `inspect_pond` 행동을 하면 정확한 값을 알 수 있지만, 그날 어획 상한이 절반이 된다 (확인에는 비용이 든다).

### 경제
| 항목 | 값 |
|---|---|
| 시작 돈 | 20 |
| 물고기 1마리 판매가 | 1 |
| 하루 생활비 | 5 |
| 1인 하루 어획 상한 | 20 |
| 파산 | 돈 < 0 상태가 **3일 연속**이면 마을을 떠남(= 사망 처리) |

## 2. 에이전트

```python
Agent:
  id: str
  name: str
  persona: str          # 한두 문장 (예: "가족이 많아 돈이 급한 어부")
  money: int
  alive: bool
  debt_days: int        # 돈 < 0 연속 일수
  memory: list[MemoryItem]
  beliefs: Beliefs      # 매일 저녁 일기에서 구조화해 기록
  trust: dict[str, float]  # 다른 에이전트 신뢰도 0~1, 일기에서 갱신
```

```python
Beliefs:
  estimated_stock: int      # 연못에 몇 마리 있다고 생각하는지
  confidence: float         # 0~1
  sustainable_guess: int    # 1인당 몇 마리까지 괜찮다고 생각하는지
  heard_rumor: bool         # (2단계) 소문을 들었는지
  believes_rumor: float     # (2단계) 소문을 얼마나 믿는지 0~1
```

- 페르소나 5개는 `configs/base.yaml`에 고정한다. 조건 간 비교를 위해 실험 내내 같은 페르소나를 쓴다.

## 3. 하루 진행 (페이즈)

LLM 호출: 에이전트당 하루 **3회** (광장, 어획, 일기). 투표가 있는 날은 +1회.

| 순서 | 페이즈 | 내용 | LLM |
|---|---|---|---|
| 1 | 아침 공지 | 날짜, 연못 추정치, 전날 총 어획량, 현재 규칙 공개 | ✗ |
| 2 | 광장 | 각자 `say` 1회 (전체 공개 or 1명에게 귓속말). 순서는 매일 시드로 섞음 | ✓ |
| 3 | 어획 | 각자 비공개로 `fish(amount)` 또는 `inspect_pond` 선택 | ✓ |
| 4 | 정산 | 규칙 엔진: 어획 배분 → 판매 → 생활비 → 파산 판정 → 붕괴 판정 → 재생 | ✗ |
| 5 | 회의 | (조건 C만) 3일마다 규칙 제안·투표 | ✓ |
| 6 | 일기 | 오늘 요약 + `beliefs`, `trust` 갱신 (JSON) | ✓ |

### 어획 배분 규칙
- 요청 합계 ≤ stock: 요청대로 지급
- 요청 합계 > stock: 요청 비율로 나눔 (내림), 남는 건 버림 (내림 때문에 남은 물고기는 아무에게도 주지 않고 연못에 그대로 남는다)
- 쿼터 규칙이 있으면: 쿼터 초과분도 일단 잡히지만 **위반**으로 기록. 처벌은 조건에 따라 다름 (EXPERIMENT 참고)

## 4. 행동 (도구 스키마)

LLM은 반드시 아래 JSON 하나로 답한다.

```json
// 광장
{"action": "say", "to": "all" | "<agent_id>", "message": "...", "reason": "..."}
// 어획
{"action": "fish", "amount": 0~20, "reason": "..."}
{"action": "inspect_pond", "reason": "..."}
// 회의
{"action": "propose_rule", "quota_per_person": int, "reason": "..."}
{"action": "vote", "proposal_id": str, "vote": "yes" | "no", "reason": "..."}
// 일기
{"summary": "...", "beliefs": {...}, "trust": {"<id>": 0.0~1.0}}
```

- `reason`은 분석용이다. 다른 에이전트에게는 보이지 않는다.
- 범위를 벗어난 값은 잘라낸다 (예: amount 30 → 20) 그리고 `clamped`로 기록한다.

## 5. 기억

- 원본 기억: 그날 본 공지, 들은 말, 내 행동과 결과
- 프롬프트에는 **최근 2일 원본 + 이전 날짜는 일기 요약만** 넣는다 (토큰 절약)
- 2단계 소문은 기억에 `"[소문] <내용> (출처: <누구>)"` 형태로 들어간다

## 6. 소문 (2단계)

```yaml
rumor:
  day: 5                # 주입하는 날
  target: "agent_03"    # 처음 듣는 에이전트
  content: "연못 물고기가 거의 바닥났대. 며칠 안에 다 사라진다더라."
  truth: false          # 실제 상태와 맞는지
```
- 주입은 해당 에이전트의 아침 기억에 "누군가에게 들은 말"로 넣는다.
- 확산 추적: 모든 `say` 메시지에 대해 소문 관련 여부를 판정한다.
  - 1차: 키워드 매칭 (`바닥`, `고갈`, `사라`)
  - 2차: 애매하면 저가 LLM 판정 호출 1회 (판정 결과도 로그)

## 7. 로그 (SQLite)

| 테이블 | 주요 컬럼 |
|---|---|
| `runs` | run_id, config_name, condition, seed, model, started_at, status |
| `days` | run_id, day, stock_start, stock_end, total_catch, collapsed |
| `agent_days` | run_id, day, agent_id, money, alive, catch, violated, beliefs_json, trust_json |
| `messages` | run_id, day, from, to, text, is_rumor |
| `actions` | run_id, day, phase, agent_id, action_json, reason, parse_error, clamped |
| `votes` | run_id, day, proposal_id, agent_id, vote |
| `llm_calls` | run_id, day, agent_id, phase, model, prompt_tokens, completion_tokens, latency_ms, cost_usd |

## 8. 종료 조건
- 설정한 일수 도달 (기본 30일)
- 연못 붕괴
- 생존 에이전트 0명
- 비용 상한 초과 (`status = aborted_cost`)
