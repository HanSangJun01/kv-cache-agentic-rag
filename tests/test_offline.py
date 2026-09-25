"""LLM·네트워크 없이 도는 결정론 검사: 인용 변환, 수치-원문 대조, 검토 규칙, 분기. 실행: python -m pytest tests 또는 python tests/test_offline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report import finalize_citations, ref_map, restore_tags, route_review, rule_check  # noqa: E402
from rag.pipeline import PAPER_TAG, key_numbers, unsupported_numbers  # noqa: E402

SOURCES = [
    {"id": "P-SW", "kind": "paper", "citation": "DeepSeek-AI(2024). DeepSeek-V2. arXiv, 2405.04434."},
    {"id": "WM1", "kind": "web", "citation": "vLLM(2025-01-01). MLA support. vLLM Blog, https://x"},
    {"id": "WD9", "kind": "web", "citation": "unused(n.d.). unused. x, https://y"},
]
PAGES = {("P-SW", 1): "reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times."}
page_text = lambda sid, p: PAGES.get((sid, p))


def test_citations_roundtrip():
    draft = "## SUMMARY\nMLA는 KV cache 93.3% 감소를 보고했다[P-SW p.1]. vLLM 지원[WM1]. 가짜[WM7].\n## REFERENCE\n옛 목록"
    md = finalize_citations(draft, SOURCES)
    assert "[1, p.1]" in md and "[2]" in md and "[WM7]" in md  # 미등록 태그는 남김
    assert "unused" not in md and "옛 목록" not in md  # 인용 안 된 출처·LLM 이 쓴 REFERENCE 제외
    assert {n: s["id"] for n, s in ref_map(md, SOURCES).items()} == {1: "P-SW", 2: "WM1"}
    assert "[P-SW p.1]" in restore_tags(md, SOURCES) and "[WM1]" in restore_tags(md, SOURCES)


def test_number_check():
    assert key_numbers("2024년 4장, 93.3%, 128K, 5.76배") == ["93.3", "128", "5.76"]
    ok = "KV cache 93.3% 감소, 처리량 5.76배[P-SW p.1]."
    bad = "처리량 6.2배[P-SW p.1], 웹 수치 42.0%[WM1]."
    look = lambda m: page_text(m[1], int(m[2]))
    assert unsupported_numbers(ok, PAPER_TAG, look) == []
    issues = unsupported_numbers(bad, PAPER_TAG, look)
    assert len(issues) == 1 and "6.2" in issues[0]  # 웹 인용 수치는 논문 대조 대상 아님
    assert unsupported_numbers("표준 지원[12][15][P-SW p.1].", PAPER_TAG, look) == []  # 인용 번호 자체는 수치 아님


def test_rule_check_and_route():
    body = "\n".join(f"## {s}\n내용" for s in ["SUMMARY", "1. 분석 배경", "2. 기술 선정", "3. 기술 개요", "4. 관점별 평가", "5. 시사점", "6. 한계점"])
    good = finalize_citations(body + "\n수치 5.76배[P-SW p.1].", SOURCES)
    critical, pages = rule_check(good, SOURCES, page_text)
    assert critical == [] and pages == 1, critical
    bad = finalize_citations(body.replace("## 5. 시사점", "## 5. 결론") + "\nMLA가 더 우수하다. 7.7배[P-SW p.1] [WM3]", SOURCES)
    critical, _ = rule_check(bad, SOURCES, page_text)
    joined = " ".join(critical)
    assert "시사점" in joined and "더 우수" in joined and "7.7" in joined and "[WM3]" in joined, critical
    assert route_review({"review": {"passed": True}}) == "approve"
    assert route_review({"review": {"passed": False}, "revision_count": 1}) == "revise"
    assert route_review({"review": {"passed": False}, "revision_count": 2}) == "unverified"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
