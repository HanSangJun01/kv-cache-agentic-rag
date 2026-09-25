"""KV cache 최적화 기술 다관점(시장성·도메인 적용) 평가 — LangGraph Multi-Agent + Agentic RAG.
실행: python app.py  →  outputs/ 에 보고서 MD·PDF, review.md, trace.json, graph.mmd 생성."""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------- 입력 (Human 기반 기술 선정: Doc Pool 에서 진영별 1개 직접 선택)
TECHNOLOGIES = {
    "sw": {
        "name": "DeepSeek-V2 MLA", "camp": "SW · 어텐션 아키텍처 개선(데이터를 작게)", "source_id": "P-SW",
        "paper": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        "file": "2405.04434.pdf", "url": "https://arxiv.org/abs/2405.04434", "date": "2024",
        "citation": "DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434.",
        "search": "DeepSeek Multi-head Latent Attention MLA", "market": "LLM inference serving",
        "reason": "KV cache 를 사후 압축하지 않고 어텐션 구조(Multi-head Latent Attention) 자체를 바꿔 KV cache 93.3% 감소를 보고한 기술. "
                  "공개 모델·오픈소스 서빙 스택 채택 사례가 있어 시장성 관점의 근거가 풍부하고, 재학습이 필요한 구조 변경이라 도메인 도입 관점과 대비가 뚜렷함.",
    },
    "hw": {
        "name": "ITME (CXL-Hybrid 계층 메모리 확장)", "camp": "HW · 메모리/스토리지 계층 확장(공간을 넓게)", "source_id": "P-HW",
        "paper": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
        "file": "2606.12556.pdf", "url": "https://arxiv.org/abs/2606.12556", "date": "2026",
        "citation": "Jang, H. et al.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556.",
        "search": "CXL memory expansion LLM inference KV cache", "market": "CXL memory",
        "reason": "원문 KV 를 그대로 두고 CXL-Hybrid 메모리로 TB 급 바이트 주소 공간을 확장하는 방식으로, 배경에서 제시된 'CXL 메모리 방식' 진영을 대표하며 "
                  "데이터센터 공유 컨텍스트 인프라를 직접 겨냥함. 2026년 발표된 메모리 기업(SK hynix) 연구라 시장 근거가 간접적이라는 점 자체가 관점 간 차이를 드러냄.",
    },
}
DOMAIN = {"name": "데이터센터·클라우드 LLM 서빙",
          "description": "대규모 동시 요청을 처리하는 클라우드 추론 서빙. GPU·HBM 비용, 처리량·지연(SLO), 품질 유지, 기존 서빙 스택 호환성이 핵심."}
AUTHOR = {"campus": "판교", "class": "9반", "name": "한상준"}  # 파일명 RAG-Output_{캠퍼스}_{X반}_{이름}.pdf
MAX_REPORT_REVISION = 2
RECURSION_LIMIT = 25


class State(TypedDict, total=False):
    technologies: dict
    domain: dict
    tech_profiles: dict
    market_eval: dict
    domain_eval: dict
    synthesis: dict
    report_md: str
    revision_count: int
    review: dict
    sources: Annotated[list[dict], operator.add]  # 병렬 노드가 함께 쓰는 키 → 누적 reducer
    trace: Annotated[list[dict], operator.add]
    outputs: dict


def build_graph():
    from agents.report import report_reviewer, report_writer, route_review, save_outputs, synthesis
    from agents.research import domain_eval, market_eval, tech_research

    g = StateGraph(State)
    g.add_node("tech_research", tech_research)
    g.add_node("market_eval", market_eval)
    g.add_node("domain_eval", domain_eval)
    g.add_node("synthesis", synthesis)
    g.add_node("report_writer", report_writer)
    g.add_node("report_reviewer", report_reviewer)
    g.add_node("save_outputs", save_outputs)
    g.add_edge(START, "tech_research")
    g.add_edge("tech_research", "market_eval")  # fan-out
    g.add_edge("tech_research", "domain_eval")
    g.add_edge(["market_eval", "domain_eval"], "synthesis")  # fan-in
    g.add_edge("synthesis", "report_writer")
    g.add_edge("report_writer", "report_reviewer")
    g.add_conditional_edges("report_reviewer", route_review,
                            {"revise": "report_writer", "approve": "save_outputs", "unverified": "save_outputs"})
    g.add_edge("save_outputs", END)
    return g.compile()


def main():
    from rag.pipeline import ROOT
    app = build_graph()
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/graph.mmd").write_text(app.get_graph().draw_mermaid())
    state = {"technologies": TECHNOLOGIES, "domain": DOMAIN, "sources": [], "trace": []}
    for step in app.stream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in step.items():
            print(f"[{node}] done -> {', '.join(update or {})}")
            if node == "report_reviewer":
                r = update["review"]
                print(f"   review passed={r['passed']} pages={r['pages']} critical={len(r['critical'])} minor={len(r['minor'])}")
            if node == "save_outputs":
                print("   " + "\n   ".join(f"{k}: {v}" for k, v in update["outputs"].items()))


if __name__ == "__main__":
    main()
