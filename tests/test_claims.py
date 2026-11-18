"""claims.py 테스트: 숫자 주장 뽑기, 기록과 견주기(±2 어림), 출처 찾기, 기록 파일에 쓰기, 보고서."""

import json
import sqlite3

from pondvillage import claims as cl
from pondvillage import logger as lg

SOONJA = "오늘은 신중하게 시작합시다. 어esterday 모두가 15마리를 잡았는데, 그럼에도 연못이 무너지지 않았으니 아직 괜찮을 것 같습니다."


def kinds(text: str) -> list[tuple[str, int]]:
    return [(c.kind, c.claimed) for c in cl.extract_claims(text)]


# ---------------------------------------------------------------- 1. 뽑기


def test_soonja_case_is_each_catch_yesterday():
    assert kinds(SOONJA) == [(cl.KIND_EACH, 15)]


def test_proposals_rules_and_limits_are_not_claims():
    for text in [
        "모두가 5마리씩 잡아보는 게 어떨까요?",
        "5마리 미만으로 줄어들면 즉시 중단해야 합니다.",
        "물고기가 최대 100마리로 두 배로 늘었을 가능성이 큽니다.",
        "20마리까지는 안전하게 잡아서 팔자.",
        "오늘은 10마리 정도만 조심하면서 잡아보는 게 어떨까요?",
        "제가 약 5마리 정도 잡아 봅니다.",
        "남은 어획량은 44마리 정도가 필요해 보이네요.",
        "물고기 수를 5마리 이상으로 유지하면서 조심해야 해.",
    ]:
        assert kinds(text) == [], text


def test_who_caught_yesterday():
    assert kinds("어제 50마리를 잡았으니, 오늘은 8마리만 잡자.") == [(cl.KIND_VILLAGE, 50)]
    assert kinds("제가 10마리를 잡았으니, 나머지는 천천히.") == [(cl.KIND_MY, 10)]
    assert kinds("어제 마을 전체 어획 29마리였다.") == [(cl.KIND_VILLAGE, 29)]
    assert kinds("어제 15마리 잡았으니 괜찮다.") == [(cl.KIND_UNCLEAR, 15)]
    assert kinds("어제 모두 5마리씩 잡았다.") == [(cl.KIND_EACH, 5)]


def test_catch_noun_and_today_after_yesterday():
    assert kinds("어제 어획량 80마리라는 정보와 오늘 연못 재고 37마리라는 점을 고려하자.") == [
        (cl.KIND_VILLAGE, 80), (cl.KIND_STOCK, 37),
    ]
    assert kinds("오늘 연못 재고가 37마리로 줄어들고 어획량도 80마리로 줄었습니다.") == [
        (cl.KIND_STOCK, 37), (cl.KIND_VILLAGE, 80),
    ]


def test_stock_claims():
    assert kinds("오늘 연못 재고가 111마리나 된다니 행운입니다.") == [(cl.KIND_STOCK, 111)]
    assert kinds("오늘은 물고기가 11마리밖에 안 남아 있다고 하니 조심하자.") == [(cl.KIND_STOCK, 11)]
    assert kinds("약 79마리일 가능성도 높아 보입니다.") == [(cl.KIND_STOCK, 79)]
    assert [c.approx for c in cl.extract_claims("연못에 약 85마리 정도 있는 것 같아")] == [True]
    assert [c.approx for c in cl.extract_claims("연못에 99마리가 있네요")] == [False]


def test_guess_reasoned_from_yesterday_catch_is_stock_not_catch():
    # P9 준비, X-qwen3gemma2 s8: 공지 81을 옮긴 말이 "어제 마을 어획 81"로 잡혀 틀린 말 42개가 됐다
    assert kinds("오늘은 어제 어획량을 고려해서 물고기가 약 81마리 있을 것 같아요.") == [(cl.KIND_STOCK, 81)]
    assert kinds("어제 어획량 30마리를 고려하면 81마리 정도는 될 것 같군.") == [(cl.KIND_VILLAGE, 30)]
    # X s8 3일째, O-gemma4qwen1 s6 6일째: 어획 숫자 뒤의 두 번째 숫자는 재고 짐작
    assert kinds("오늘은 어제 어획량이 40마리라서 연못에 약 97마리가 있을 것 같아.") == [
        (cl.KIND_VILLAGE, 40), (cl.KIND_STOCK, 97)]
    assert kinds("여러분, 어제 어획량이 25마리로 줄어 재고가 85마리 정도로 추정됩니다.") == [
        (cl.KIND_VILLAGE, 25), (cl.KIND_STOCK, 85)]


def test_future_and_hypothetical_numbers_are_not_stock_claims():
    # P9 준비, S-qwen s10·X-qwen3gemma2 s7: 앞날 계산·가정이 "지어낸 재고"로 잡혔다
    for text in [
        "남은 물고기가 83마리니까, 40마리가 남을 거에요.",
        "그렇다면 2일째에 물고기가 86마리가 될 거에요.",
        "남은 돈 5원을 연못에 투자해서 물고기 1마리 더 늘리는 것을 목표로 하겠습니다.",
        "그 돈으로 연못에 6마리 물고기 투자해서 103마리로 만들자.",
    ]:
        assert all(c.claimed not in (40, 86, 1, 6, 103) for c in cl.extract_claims(text)), text
    assert kinds("연못에 40마리가 남아 있어요.") == [(cl.KIND_STOCK, 40)]  # 지금 남은 양은 그대로 잡는다


def test_extract_never_crashes():
    assert cl.extract_claims("") == []
    assert cl.extract_claims(None) == []  # type: ignore[arg-type]
    assert cl.extract_claims("羟一起努力吧 ddie 9999") == []


# ---------------------------------------------------------------- 2. 견주기

FACTS = cl.DayFacts(
    stock_start={1: 100, 2: 100, 3: 50},
    total_catch={1: 29, 2: 75},
    catches={1: {"a1": 5, "a2": 5, "a3": 7}, 2: {"a1": 15, "a2": 15}},
)


def claim(kind, n, approx=False):
    return cl.Claim(kind, n, approx, "")


def test_judge_each_and_my_and_village():
    assert cl.judge(claim(cl.KIND_EACH, 15), 2, "a2", FACTS) == ("5~7", 1)
    assert cl.judge(claim(cl.KIND_EACH, 15), 3, "a2", FACTS) == ("15", 0)
    assert cl.judge(claim(cl.KIND_MY, 7), 2, "a3", FACTS) == ("7", 0)
    assert cl.judge(claim(cl.KIND_VILLAGE, 30), 2, "a1", FACTS) == ("29", 1)


def test_approx_tolerance_is_two():
    assert cl.judge(claim(cl.KIND_STOCK, 98, approx=True), 1, "a1", FACTS) == ("100", 0)
    assert cl.judge(claim(cl.KIND_STOCK, 97, approx=True), 1, "a1", FACTS) == ("100", 1)
    assert cl.judge(claim(cl.KIND_STOCK, 98), 1, "a1", FACTS) == ("100", 1)


def test_judge_edge_cases():
    assert cl.judge(claim(cl.KIND_VILLAGE, 30), 1, "a1", FACTS) == ("어제 없음", 1)
    assert cl.judge(claim(cl.KIND_UNCLEAR, 15), 2, "a1", FACTS) == (None, None)
    assert cl.judge(claim(cl.KIND_STOCK, 50), 9, "a1", FACTS) == (None, None)


# ---------------------------------------------------------------- 3. 출처

USER = """너의 이름은 순자.
[지금까지의 요약 (1일째 일기)]
오늘은 아침부터 모두가 동의하여 15마리를 잡았습니다.
[최근 기록]
1일째: 아침 추정치 약 111마리. 5마리를 원했고 5마리를 잡았다. 판 돈 5원, 생활비 5원, 남은 돈 20원.
1일째 광장: 영철(모두에게): "연못에 42마리 있대" / 나(모두에게): "13마리만 잡자"
[지난 일기에 적은 내 생각]
연못에 약 50마리가 있을 것 같다(확신 0.7). 한 사람이 하루 3마리까지는 괜찮을 것 같다.
오늘은 2일째다.
아침 공지: 연못 재고 추정치 약 93마리. 어제 마을 전체 어획 29마리.
내 돈: 20원.
이웃: 만석, 영철.
[지금까지 광장에서 들은 말]
만석(모두에게): "어제 모두 21마리 잡았지"
광장에서 할 말을 JSON으로 답하라.
"""


def test_prompt_pools_reads_each_section():
    pools = cl.prompt_pools(USER)
    assert {15, 5, 29, 50, 3, 13} <= pools.memory  # 요약("오늘은 …"으로 시작해도), 사실, 지난 생각, 내가 한 말
    assert 111 not in pools.memory and 93 not in pools.memory
    assert pools.notice == {111, 93} and pools.today_estimate == 93
    assert pools.heard == {42, 21}


def test_source_order():
    pools = cl.prompt_pools(USER)
    assert cl.find_source(15, pools) == cl.SRC_MEMORY
    assert cl.find_source(93, pools, cl.KIND_STOCK) == cl.SRC_NOTICE
    assert cl.find_source(21, pools) == cl.SRC_HEARD
    assert cl.find_source(77, pools) == cl.SRC_IMPROVISED
    assert cl.find_source(77, None) is None
    pools.memory.add(93)  # 일기에도 있지만 그날 공지와 같은 재고 말이면 공지
    assert cl.find_source(93, pools, cl.KIND_STOCK) == cl.SRC_NOTICE
    assert cl.find_source(93, pools, cl.KIND_VILLAGE) == cl.SRC_MEMORY


def test_broken_prompt_is_empty():
    assert cl.prompt_user_text("not json") == ""
    assert cl.prompt_user_text(None) == ""
    assert cl.prompt_pools("").memory == set()


# ---------------------------------------------------------------- 기록 파일에 돌리기


def make_db(path, messages, run_id="r_s3", condition="llm-same"):
    conn = lg.open_db(path)
    conn.execute("INSERT INTO runs VALUES (?, 'base', ?, 3, 'qwen2.5:7b', 't', 'collapsed')", (run_id, condition))
    for day, s, e, t in [(1, 100, 100, 29), (2, 100, 50, 75)]:
        conn.execute("INSERT INTO days VALUES (?, ?, ?, ?, ?, 0)", (run_id, day, s, e, t))
    for day, agent, c in [(1, "a1", 5), (1, "a2", 5), (1, "a3", 7), (2, "a1", 15), (2, "a2", 15)]:
        conn.execute("INSERT INTO agent_days VALUES (?, ?, ?, 20, 1, ?, 0, NULL, NULL)", (run_id, day, agent, c))
    prompt = json.dumps([{"role": "system", "content": "규칙 100마리"}, {"role": "user", "content": USER}],
                        ensure_ascii=False)
    for day, frm, to, text in messages:
        conn.execute('INSERT INTO messages (run_id, day, "from", "to", text) VALUES (?, ?, ?, ?, ?)',
                     (run_id, day, frm, to, text))
        conn.execute(
            "INSERT INTO llm_calls (run_id, day, agent_id, phase, model, prompt) VALUES (?, ?, ?, 'say', 'qwen2.5:7b', ?)",
            (run_id, day, frm, prompt),
        )
    conn.commit()
    conn.close()


def test_collect_writes_claims_and_is_repeatable(tmp_path):
    db = tmp_path / "a.db"
    make_db(db, [(2, "a2", "all", SOONJA), (2, "a1", "all", "모두 5마리씩 잡자")])
    for _ in range(2):
        [res] = cl.collect([db], write=True)
    rows = sqlite3.connect(db).execute(
        "SELECT day, agent_id, kind, claimed, actual, is_false, source, judged_by FROM claims"
    ).fetchall()
    assert rows == [(2, "a2", cl.KIND_EACH, 15, "5~7", 1, cl.SRC_MEMORY, "code")]
    assert res.messages == 2 and not res.solo


def test_collect_without_write_leaves_file_alone(tmp_path):
    db = tmp_path / "a.db"
    make_db(db, [(2, "a2", "all", SOONJA)])
    cl.collect([db])
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM claims").fetchone() == (0,)


def test_report_sample_and_cli(tmp_path):
    make_db(tmp_path / "a.db", [(2, "a2", "all", SOONJA), (2, "a1", "all", "연못에 약 93마리 있대요")])
    make_db(tmp_path / "solo.db", [(2, "a1", "self", "오늘도 20마리 잡아야지")], run_id="r2", condition="solo")
    (tmp_path / "empty.db").write_bytes(b"")
    assert cl.main(["--runs-dir", str(tmp_path), "--sample-size", "5"]) == 0
    report = (tmp_path / "claims_report.md").read_text(encoding="utf-8")
    assert "틀린 주장 1개" in report and "공지 추정치를 옮겨 실제 재고와 다른 말 1개" in report
    assert "(혼자 말하기)" in report and "**통과**" in report
    sample = (tmp_path / "claims_sample.csv").read_text(encoding="utf-8-sig")
    assert "어제 각자 어획 15→5~7 틀림 memory" in sample


def test_cli_without_talk_runs(tmp_path):
    lg.open_db(tmp_path / "x.db").close()
    assert cl.main(["--runs-dir", str(tmp_path)]) == 1


def _res(cond, seed, n_false, solo=False):
    rows = [cl.ClaimRow("r", 3, i, "agent_01", "all", "약 79마리", cl.KIND_STOCK, 79, "100", 1, cl.SRC_MEMORY, "q")
            for i in range(n_false)]
    from collections import Counter
    return cl.RunResult(f"r{seed}", cond, seed, "q", 10, solo, rows, Counter(), [])


def test_false_distribution_buckets_and_top_share():
    results = [_res("exp1-S-qwen", s, n) for s, n in [(1, 0), (2, 13), (3, 1), (4, 2), (5, 0)]]
    results += [_res("exp1-solo-qwen", 1, 0, solo=True)]
    s_qwen, solo = cl.false_distribution(results)
    assert (s_qwen["runs"], s_qwen["zero"], s_qwen["few"], s_qwen["many"], s_qwen["total"]) == (5, 2, 2, 1, 16)
    assert (s_qwen["top_seed"], s_qwen["top"]) == (2, 13) and s_qwen["top_share"] == 13 / 16
    assert solo["solo"] and solo["total"] == 0
    text = cl.report_markdown(results)
    assert "마을 종류별 분포" in text and "s2 13개 (81%)" in text
    assert "전체 16개 중 13개(81%)" in text
