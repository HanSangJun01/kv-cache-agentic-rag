"""RAG 에이전트 3종: 🔍 기술 조사 / 📊 시장성 평가 / 🏭 도메인 평가.
각 에이전트는 기술 2건(SW ∥ HW)을 노드 내부 스레드로 독립 처리하고, 끝에 Reflection(근거 재대조)을 돈다."""
import itertools
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from prompts.criteria import CRITERIA, PROFILE_QUESTIONS, criteria_table
from prompts.templates import COMMON_RULES, EVAL_PROMPT, FIX_PROMPT, PROFILE_PROMPT, REFLECT_PROMPT
from rag.pipeline import PAPER_TAG, agentic_retrieve, generator, judge, knowledge_base, unsupported_numbers, web_search, web_source

ANY_TAG = re.compile(r"\[(?:P-SW|P-HW) p\.\d+\]|\[W[MD]\d+\]")


# ---------------------------------------------------------------- 구조화 출력 스키마
class TechProfile(BaseModel):
    overview: str = Field(description="기술 개요 (출처 태그 포함)")
    core_method: str = Field(description="핵심 방식")
    reported_results: list[str] = Field(description="저자 보고 성과: 수치·지표·조건 + 태그")
    experimental_setup: str = Field(description="실험 조건: 모델·HW·baseline")
    limitations: list[str] = Field(description="저자가 밝힌 한계·오버헤드·전제")
    deployment_requirements: list[str] = Field(description="도입 조건")


class CriterionAssessment(BaseModel):
    criterion_id: str
    rating: Literal["높음", "중간", "낮음", "판단 유보"]
    confidence: Literal["높음", "중간", "낮음"]
    rationale: str = Field(description="등급 근거 2~3문장, 출처 태그 포함")
    supporting_evidence: list[str] = Field(description="지지 근거 (출처 태그 필수)")
    counter_evidence: list[str] = Field(description="반대·한계·리스크 근거. 없으면 '확인된 반대 근거 없음'")


class PerspectiveEvaluation(BaseModel):
    assessments: list[CriterionAssessment]
    overall_view: str = Field(description="이 관점에서 기술이 어떻게 인식되는가")
    information_gaps: list[str]


class Reflection(BaseModel):
    issues: list[str]


# ---------------------------------------------------------------- 공통
def dump(obj) -> str:
    return json.dumps(obj.model_dump() if isinstance(obj, BaseModel) else obj, ensure_ascii=False, indent=1)


def evidence_text(evidence: list[dict]) -> str:
    return "\n\n".join(f"{e['tag']} {e.get('label', '')}\n{e['text']}" for e in evidence) or "(근거 없음)"


def dedupe(evidence: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in evidence:
        if (e["tag"], e["text"]) not in seen:
            seen.add((e["tag"], e["text"]))
            out.append(e)
    return out


def reflect(kb, schema, obj: BaseModel, evidence: list[dict], name: str, tech: str, rubric: str = "") -> tuple[BaseModel, dict]:
    """Reflection: 결과를 인용 페이지 원문과 재대조. 결정론(태그·수치) + Judge LLM → 문제 있으면 1회 수정."""
    text = dump(obj)
    allowed = {e["tag"] for e in evidence}
    issues = [f"근거 목록에 없는 출처 태그 {t}" for t in sorted(set(ANY_TAG.findall(text)) - allowed)]
    issues += unsupported_numbers(text, PAPER_TAG, lambda m: kb.page_text(m[1], int(m[2])))
    cited = sorted(set(ANY_TAG.findall(text)) & allowed)
    web = {e["tag"]: e["text"] for e in evidence if e["tag"].startswith("[W")}
    sources = "\n\n".join(f"{t}\n{web[t] if t in web else kb.page_text(*_page(t))}" for t in cited)
    issues += judge().with_structured_output(Reflection).invoke(REFLECT_PROMPT.format(name=name, rubric=rubric and f"평가 루브릭:\n{rubric}\n", sources=sources, output=text)).issues
    if issues:
        obj = generator().with_structured_output(schema).invoke(FIX_PROMPT.format(
            rules=COMMON_RULES + (f"\n평가 루브릭:\n{rubric}" if rubric else ""), issues="\n".join(f"- {i}" for i in issues), evidence=evidence_text(evidence), output=text))
    return obj, {"agent_tech": tech, "reflection": name, "issues": issues, "revised": bool(issues)}


def _page(tag: str) -> tuple[str, int]:
    m = PAPER_TAG.fullmatch(tag)
    return m[1], int(m[2])


def per_tech(fn, techs: dict) -> dict:
    """기술 간 병렬(SW ∥ HW) — 그래프 노드를 늘리지 않고 노드 내부에서 처리."""
    with ThreadPoolExecutor(len(techs)) as ex:
        return dict(zip(techs, ex.map(fn, techs)))


# ---------------------------------------------------------------- 🔍 기술 조사 에이전트
def tech_research(state: dict) -> dict:
    techs = state["technologies"]
    kb = knowledge_base(techs)

    def run(tech):
        t = techs[tech]
        with ThreadPoolExecutor(4) as ex:
            results = list(ex.map(lambda q: agentic_retrieve(kb, q.format(name=t["name"]), tech, t["name"]), PROFILE_QUESTIONS))
        evidence = dedupe([e for ev, _ in results for e in ev])
        profile = generator().with_structured_output(TechProfile).invoke(
            PROFILE_PROMPT.format(name=t["name"], rules=COMMON_RULES, evidence=evidence_text(evidence)))
        profile, rtrace = reflect(kb, TechProfile, profile, evidence, t["name"], tech)
        return profile.model_dump(), [tr for _, tr in results] + [rtrace]

    out = per_tech(run, techs)
    sources = [{"id": t["source_id"], "kind": "paper", "ref_type": "논문", "tech": tech, "title": t["paper"], "url": t["url"], "site": "arXiv",
                "date": t["date"], "citation": t["citation"]} for tech, t in techs.items()]
    return {"tech_profiles": {k: v[0] for k, v in out.items()}, "sources": sources,
            "trace": [{"node": "tech_research", **tr} for v in out.values() for tr in v[1]]}


# ---------------------------------------------------------------- 📊 시장성 · 🏭 도메인 평가 에이전트 (같은 절차, 다른 기준·판단 책임)
def evaluate_perspective(state: dict, perspective: str, prefix: str) -> tuple[dict, list, list]:
    techs, domain = state["technologies"], state["domain"]
    kb = knowledge_base(techs)
    counter, lock, url_ids, sources = itertools.count(1), threading.Lock(), {}, []

    def web(query, tech, cid):
        evidence, ids = [], []
        for r in web_search(query):
            with lock:  # 노드별 prefix(WM/WD) + URL 중복 제거
                new = r["url"] not in url_ids
                if new:
                    url_ids[r["url"]] = f"{prefix}{next(counter)}"
            sid = url_ids[r["url"]]
            src = web_source(r, sid, tech)
            if new:
                sources.append({k: v for k, v in src.items() if k != "text"})
            evidence.append({"tag": f"[{sid}]", "label": f"(웹, {cid}) {src['citation']}", "text": src["text"]})
            ids.append(sid)
        return evidence, {"agent_tech": tech, "criterion": cid, "web_query": query, "results": ids}

    def criterion(tech, c):
        t = techs[tech]
        evidence, traces = [], []
        if c["rag_q"]:
            ev, tr = agentic_retrieve(kb, c["rag_q"].format(name=t["name"]), tech, t["name"])
            evidence += [{**e, "label": f"(논문, {c['id']})"} for e in ev]
            traces.append({**tr, "criterion": c["id"]})
        queries = c["web_q"] or ()
        if c["rag_q"] and not evidence and not queries:  # 논문에 근거 없음 → 웹 fallback
            queries = (c["rag_q"].format(name=t["search"]), f"{t['search']} limitations overhead")
        for q in queries:
            ev, tr = web(q.format(search=t["search"], market=t["market"]), tech, c["id"])
            evidence += ev
            traces.append(tr)
        return evidence, traces

    def run(tech):
        t = techs[tech]
        with ThreadPoolExecutor(4) as ex:
            results = list(ex.map(lambda c: criterion(tech, c), CRITERIA[perspective]))
        evidence = dedupe([e for ev, _ in results for e in ev])
        name = "시장성" if perspective == "market" else "도메인 적용"
        result = generator().with_structured_output(PerspectiveEvaluation).invoke(EVAL_PROMPT.format(
            perspective_name=name, domain=f"{domain['name']} — {domain['description']}", name=t["name"], camp=t["camp"],
            criteria=criteria_table(perspective), rules=COMMON_RULES, profile=dump(state["tech_profiles"][tech]),
            evidence=evidence_text(evidence)))
        result, rtrace = reflect(kb, PerspectiveEvaluation, result, evidence, t["name"], tech, criteria_table(perspective))
        return result.model_dump(), [tr for _, trs in results for tr in trs] + [rtrace]

    out = per_tech(run, techs)
    node = f"{perspective}_eval"
    return {k: v[0] for k, v in out.items()}, sources, [{"node": node, **tr} for v in out.values() for tr in v[1]]


def market_eval(state: dict) -> dict:
    result, sources, trace = evaluate_perspective(state, "market", "WM")
    return {"market_eval": result, "sources": sources, "trace": trace}


def domain_eval(state: dict) -> dict:
    result, sources, trace = evaluate_perspective(state, "domain", "WD")
    return {"domain_eval": result, "sources": sources, "trace": trace}
