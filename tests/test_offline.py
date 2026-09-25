"""LLM·네트워크 없이 도는 결정론 검사: 인용 변환, 수치-원문 대조, 검토 규칙, 분기. 실행: python -m pytest tests 또는 python tests/test_offline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report import finalize_citations, ref_map, restore_tags, route_review, rule_check  # noqa: E402
from rag.pipeline import PAPER_TAG, format_citation, key_numbers, landing_url, parse_meta, unsupported_numbers  # noqa: E402

SOURCES = [
    {"id": "P-SW", "kind": "paper", "ref_type": "논문", "citation": "DeepSeek-AI(2024). DeepSeek-V2. *arXiv*, 2405.04434."},
    {"id": "WM1", "kind": "web", "ref_type": "기타", "citation": "vLLM(2025-01-01). *MLA support*. vLLM Blog, https://x"},
    {"id": "WD9", "kind": "web", "ref_type": "기타", "citation": "unused(n.d.). *unused*. x, https://y"},
    {"id": "WD3", "kind": "web", "ref_type": "논문", "citation": "Ji, T. et al.(2025). MLA anywhere. *ACL*."},
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
    grouped = finalize_citations("## SUMMARY\n웹 먼저[WM1]. 논문 둘째[WD3]. 원문[P-SW p.1].", SOURCES)
    assert "웹 먼저[3]. 논문 둘째[1]. 원문[2, p.1]" in grouped  # 번호: 논문 → 특허 → 기타 그룹 순, 그룹 안은 첫 인용 순
    ref = grouped.split("## REFERENCE")[1]
    assert ref.index("### 논문") < ref.index("[1] Ji") < ref.index("[2] DeepSeek") < ref.index("### 기타") < ref.index("[3] vLLM")
    assert "### 특허" not in ref  # 인용이 없는 그룹은 제목도 생략


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


def test_reference_formats():
    """REFERENCE 표기 지침: 특허 / 논문 / 웹."""
    patent = parse_meta('<meta name="DC.title" content="KV Cache Transform Coding"><meta name="DC.date" content="2023-04-30" scheme="dateSubmitted">'
                        '<meta name="DC.date" content="2024-06-13"><meta name="DC.contributor" content="SK Hynix Inc" scheme="assignee">'
                        '<meta name="citation_patent_publication_number" content="US:20240193040:A1">')
    url = "https://patents.google.com/patent/US20240193040A1/en"
    assert format_citation({"title": "US20240193040A1 - KV ..."}, patent, url)[0] == \
        f"SK Hynix Inc(2024-06). *KV Cache Transform Coding*, US-20240193040-A1, {url}"
    paper = parse_meta('<meta content="Towards Economical Inference" name=citation_title><meta content="Tao Ji" name=citation_author>'
                       '<meta content="Bin Guo" name=citation_author><meta content="Proceedings of ACL" name=citation_conference_title>'
                       '<meta content="2025/7" name=citation_publication_date>')  # 따옴표 없는 속성 (ACL Anthology)
    assert format_citation({"title": "x"}, paper, "https://aclanthology.org/x")[0] == "Ji, T. et al.(2025). Towards Economical Inference. *Proceedings of ACL*."
    arxiv = parse_meta('<meta name="citation_title" content="Hardware-Centric Analysis of DeepSeek&#39;s MLA"><meta name="citation_author" content="Geens, Robin">'
                       '<meta name="citation_date" content="2025/06/03"><meta name="citation_arxiv_id" content="2506.02523">')
    assert format_citation({}, arxiv, "https://arxiv.org/abs/2506.02523")[0] == "Geens, R.(2025). Hardware-Centric Analysis of DeepSeek's MLA. *arXiv*, 2506.02523."
    web = parse_meta('<meta property="og:site_name" content="Google Research Blog"><meta property="og:title" content="TurboQuant for KV Cache Compression">'
                     '<meta property="article:published_time" content="2026-03-30T09:00:00Z"><meta name="author" content="Google Research">')
    url = "https://research.google/blog/turboquant"
    assert format_citation({"title": "TurboQuant for KV Cache ...."}, web, url)[0] == \
        f"Google Research(2026-03-30). *TurboQuant for KV Cache Compression*. Google Research Blog, {url}"  # 잘린 제목 복원
    assert format_citation({"title": "MLA Guide | Raschka"}, {"og:site_name": ["Raschka"]}, "https://r.com/a")[0] == \
        "Raschka(n.d.). *MLA Guide*. Raschka, https://r.com/a"  # 사이트명 꼬리 제거
    assert format_citation({"title": "LLM Market - Size, Share & ...."}, {}, "https://m.biz/r")[0] == \
        "m.biz(n.d.). *LLM Market - Size, Share*. m.biz, https://m.biz/r"  # 복원 불가 시 잘린 꼬리 정리
    assert [format_citation({}, m, "https://patents.google.com/patent/US1A/en" if m is patent else "https://a")[4]
            for m in (patent, paper, web)] == ["특허", "논문", "기타"]  # REFERENCE 소제목 분류
    assert landing_url("https://aclanthology.org/2025.acl-long.1597.pdf") == "https://aclanthology.org/2025.acl-long.1597"
    assert landing_url("https://arxiv.org/pdf/2506.02523v2") == "https://arxiv.org/abs/2506.02523"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
