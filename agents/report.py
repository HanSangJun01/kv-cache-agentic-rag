"""보고 단계 에이전트 3종: ⚖️ 평가 종합 / 📝 보고서 생성 / ✅ 보고서 검토, + 💾 저장 노드(에이전트 아님).
인용 태그 → 번호·REFERENCE 변환, 규칙 검사, PDF 렌더링도 여기서 처리한다."""
import json
import re
from datetime import date
from typing import Literal

import markdown
import weasyprint
from pydantic import BaseModel

from prompts.criteria import BIAS_STRATEGIES, CRITERIA, FORBIDDEN_PHRASES, MAX_PAGES, REQUIRED_SECTIONS, SUMMARY_MAX_CHARS
from prompts.templates import REPORT_PROMPT, REVIEW_PROMPT, REVISE_PROMPT, SYNTHESIS_PROMPT
from rag.pipeline import ROOT, SOURCE_TAG, generator, judge, knowledge_base, unsupported_numbers

NUM_CITE = re.compile(r"\[(\d+), p\.(\d+)\]")


# ---------------------------------------------------------------- ⚖️ 평가 종합 에이전트
class ConflictPoint(BaseModel):
    topic: str
    tech: Literal["SW", "HW", "공통"]
    market_view: str
    domain_view: str
    why_diverge: str


class Synthesis(BaseModel):
    agreements: list[str]
    conflicts: list[ConflictPoint]
    tech_contrast: list[str]
    complementarity: str
    caveats: list[str]


def synthesis(state: dict) -> dict:
    result = generator().with_structured_output(Synthesis).invoke(SYNTHESIS_PROMPT.format(
        domain=state["domain"]["name"], profiles=_j(state["tech_profiles"]), market=_j(state["market_eval"]),
        domain_eval=_j(state["domain_eval"])))
    return {"synthesis": result.model_dump()}


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 인용 태그 ⇄ 번호 · REFERENCE
REF_GROUPS = ["논문", "특허", "기타"]  # REFERENCE 표기 지침의 구분 순서


def finalize_citations(draft: str, sources: list[dict]) -> str:
    """본문 태그를 번호로 바꾸고, 실제 인용된 출처만 REFERENCE 로 만든다(논문 → 특허 → 기타 소제목, 그룹 안은 첫 인용 순).
    미등록 태그는 그대로 남겨 검토에서 잡는다."""
    by_id = {s["id"]: s for s in sources}
    body = re.split(r"\n#+\s*REFERENCE", draft)[0].rstrip()
    cited = list(dict.fromkeys(m[1] or m[3] for m in SOURCE_TAG.finditer(body) if (m[1] or m[3]) in by_id))
    group = lambda sid: by_id[sid].get("ref_type", "기타")
    ordered = [sid for g in REF_GROUPS for sid in cited if group(sid) == g]
    number = {sid: n for n, sid in enumerate(ordered, 1)}

    def sub(m):
        sid = m[1] or m[3]
        if sid not in number:
            return m[0]
        return f"[{number[sid]}, p.{m[2]}]" if m[1] else f"[{number[sid]}]"

    body = SOURCE_TAG.sub(sub, body)
    refs = "\n\n".join(f"### {g}\n\n" + "\n\n".join(f"[{number[sid]}] {by_id[sid]['citation']}" for sid in ordered if group(sid) == g)
                       for g in REF_GROUPS if any(group(sid) == g for sid in ordered))
    return f"{body}\n\n## REFERENCE\n\n{refs}\n"


def ref_map(md: str, sources: list[dict]) -> dict[int, dict]:
    by_citation = {s["citation"]: s for s in sources}
    ref = md.split("## REFERENCE", 1)[-1]
    return {int(n): by_citation[c] for n, c in re.findall(r"^\[(\d+)\] (.+)$", ref, re.M) if c in by_citation}


def restore_tags(md: str, sources: list[dict]) -> str:
    """재작성용: 번호 인용을 원래 태그로 되돌린다."""
    refs = ref_map(md, sources)
    body = md.split("## REFERENCE", 1)[0]
    body = body[body.find("## SUMMARY"):] if "## SUMMARY" in body else body
    body = NUM_CITE.sub(lambda m: f"[{refs[int(m[1])]['id']} p.{m[2]}]" if int(m[1]) in refs else m[0], body)
    return re.sub(r"\[(\d+)\]", lambda m: f"[{refs[int(m[1])]['id']}]" if int(m[1]) in refs else m[0], body)


# ---------------------------------------------------------------- 📝 보고서 생성 에이전트
def ratings_table(evals: dict, perspective: str, techs: dict) -> str:
    names = {c["id"]: c["name"] for c in CRITERIA[perspective]}
    rows = []
    for cid, cname in names.items():
        cells = []
        for tech in ("sw", "hw"):
            a = next((a for a in evals[tech]["assessments"] if a["criterion_id"] == cid), None)
            cells.append(f"{a['rating']} (신뢰도 {a['confidence']})" if a else "판단 유보")
        rows.append(f"| {cid} {cname} | {cells[0]} | {cells[1]} |")
    return f"| 기준 | SW: {techs['sw']['name']} | HW: {techs['hw']['name']} |\n|---|---|---|\n" + "\n".join(rows)


def report_writer(state: dict) -> dict:
    techs, sources, review = state["technologies"], state["sources"], state.get("review")
    prompt = REPORT_PROMPT.format(
        summary_max=SUMMARY_MAX_CHARS, domain=state["domain"]["name"],
        selection_sentence="본 평가는 에이전트 기반 선정 대신 Human 기반 방식으로, 과제 Doc Pool 에서 SW·HW 진영별 1개 기술을 직접 선정했다.",
        forbidden=", ".join(f'"{p}"' for p in FORBIDDEN_PHRASES),
        technologies="\n".join(f"- {k.upper()} ({t['camp']}): {t['name']} — 논문: {t['paper']} [{t['source_id']}] / 선정 사유: {t['reason']}" for k, t in techs.items()),
        profiles=_j(state["tech_profiles"]), market=_j(state["market_eval"]), domain_eval=_j(state["domain_eval"]),
        synthesis=_j(state["synthesis"]), market_table=ratings_table(state["market_eval"], "market", techs),
        domain_table=ratings_table(state["domain_eval"], "domain", techs),
        bias="\n".join(f"{i}. {b}" for i, b in enumerate(BIAS_STRATEGIES, 1)),
        source_tags="\n".join(f"- [{s['id']}{' p.N' if s['kind'] == 'paper' else ''}] {s['citation']}" for s in sources))
    count = 0
    if review:  # Loop: 검토 반려 → 지적 사항 반영 재작성
        count = state.get("revision_count", 0) + 1
        prompt += REVISE_PROMPT.format(critical="\n".join(f"- {c}" for c in review["critical"]) or "- 없음",
                                       minor="\n".join(f"- {c}" for c in review["minor"]) or "- 없음",
                                       draft=restore_tags(state["report_md"], sources))
    draft = generator().invoke(prompt).content
    draft = draft[draft.find("## SUMMARY"):] if "## SUMMARY" in draft else draft
    header = (f"# KV cache 최적화 기술 다관점 평가 보고서\n\n**SW 압축 vs HW 메모리 확장** — {techs['sw']['name']} · {techs['hw']['name']}  \n"
              f"평가 관점: 시장성 · 도메인 적용({state['domain']['name']}) | 작성일: {date.today().isoformat()} | 자동 생성(LangGraph Multi-Agent + Agentic RAG)\n\n")
    return {"report_md": header + finalize_citations(draft, sources), "revision_count": count}


# ---------------------------------------------------------------- ✅ 보고서 검토 에이전트 (규칙 = 결정론, Judge = LLM)
class JudgeReview(BaseModel):
    critical: list[str]
    minor: list[str]


CSS = """@page { size: A4; margin: 16mm 15mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", sans-serif; font-size: 9.5pt; line-height: 1.5; color: #111; }
h1 { font-size: 16pt; margin: 0 0 4pt; } h2 { font-size: 12.5pt; border-bottom: 1px solid #999; margin: 12pt 0 4pt; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 8pt 0 3pt; } p, li { margin: 2pt 0; } ul { padding-left: 14pt; margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 4pt 0; } th, td { border: 1px solid #aaa; padding: 2pt 4pt; vertical-align: top; }
th { background: #eee; }
em { font-style: italic; font-family: "Helvetica Neue", Arial, "Liberation Sans", "DejaVu Sans", "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", sans-serif; } blockquote { border-left: 3px solid #c00; margin: 4pt 0; padding: 2pt 8pt; background: #fff3f3; }"""


def render_pdf(md: str) -> tuple[bytes, int]:
    md = re.sub(r"\[(\d+(?:, p\.\d+)?)\]", r"&#91;\1&#93;", md)  # 인용 [1][2] 가 참조형 링크로 해석돼 표가 깨지는 것 방지
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{markdown.markdown(md, extensions=['tables'])}</body></html>"
    doc = weasyprint.HTML(string=html).render()
    return doc.write_pdf(), len(doc.pages)


def section(md: str, title: str) -> str:
    m = re.search(rf"^##\s*{re.escape(title)}[^\n]*\n(.*?)(?=^##\s|\Z)", md, re.M | re.S)
    return m.group(1) if m else ""


def plain_len(text: str) -> int:
    return len(re.sub(r"[#*|`>\-]|\[\d+(?:, p\.\d+)?\]", "", text).strip())


def leftover_tags(body: str, sources: list[dict]) -> list[str]:
    """번호로 바뀌지 않고 남은 출처 태그. 레지스트리 ID 의 접두어(P-SW, WM …)로 시작하는 괄호만 본다 → [M1] 같은 일반 괄호는 제외."""
    prefixes = sorted({re.sub(r"\d+$", "", s["id"]) for s in sources}, key=len, reverse=True)
    if not prefixes:
        return []
    return sorted(set(re.findall(r"\[(?:%s)[^\]]*\]" % "|".join(map(re.escape, prefixes)), body)))


def rule_check(md: str, sources: list[dict], page_text) -> tuple[list[str], int]:
    critical = []
    heads = re.findall(r"^##\s+(.+)$", md, re.M)
    if not heads or "SUMMARY" not in heads[0] or "REFERENCE" not in heads[-1]:
        critical.append("[규칙] 첫 장은 SUMMARY, 마지막 장은 REFERENCE 여야 함")
    critical += [f"[규칙] 필수 목차 누락: {s}" for s in REQUIRED_SECTIONS if not any(s in h for h in heads)]
    if (n := plain_len(section(md, "SUMMARY"))) > SUMMARY_MAX_CHARS:
        critical.append(f"[규칙] SUMMARY {n}자 > {SUMMARY_MAX_CHARS}자(반 페이지) — 줄일 것")
    body = md.split("## REFERENCE", 1)[0]
    critical += [f"[규칙] 우열·추천 금칙어 사용: \"{p}\"" for p in FORBIDDEN_PHRASES if p in body]
    critical += [f"[규칙] 등록되지 않았거나 형식이 틀린 출처 태그: {t}" for t in leftover_tags(body, sources)]
    refs = ref_map(md, sources)
    page_of = lambda m: page_text(refs[int(m[1])]["id"], int(m[2])) if int(m[1]) in refs and refs[int(m[1])]["kind"] == "paper" else None
    critical += [f"[규칙] {i}" for i in unsupported_numbers(body, NUM_CITE, page_of)]
    _, pages = render_pdf(md)
    if pages > MAX_PAGES:
        critical.append(f"[규칙] PDF {pages}쪽 > {MAX_PAGES}쪽 — 분량 축소 필요")
    return critical, pages


def report_reviewer(state: dict) -> dict:
    md, sources = state["report_md"], state["sources"]
    kb = knowledge_base(state["technologies"])
    critical, pages = rule_check(md, sources, kb.page_text)
    evidence = _j({"market_eval": state["market_eval"], "domain_eval": state["domain_eval"], "synthesis": state["synthesis"]})
    verdict = judge().with_structured_output(JudgeReview).invoke(REVIEW_PROMPT.format(evidence=evidence, report=md))
    critical += [f"[Judge] {c}" for c in verdict.critical]
    return {"review": {"passed": not critical, "critical": critical, "minor": verdict.minor, "pages": pages}}


def route_review(state: dict, *, max_revision: int) -> str:
    """Branch: approve / revise / unverified(재작성 한도 도달). max_revision 은 그래프 조립 시 주입."""
    if state["review"]["passed"]:
        return "approve"
    return "revise" if state.get("revision_count", 0) < max_revision else "unverified"


# ---------------------------------------------------------------- 💾 저장 노드 (단순 I/O)
def save_outputs(state: dict, *, author: dict) -> dict:
    """author(파일명용 캠퍼스·반·이름)는 그래프 조립 시 주입."""
    review, md = state["review"], state["report_md"]
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if not review["passed"]:  # 한도 도달: 무한 루프 대신 미통과 표시 후 저장
        warn = f"> ⚠️ **검토 미통과(unverified)** — 재작성 {state.get('revision_count', 0)}회 후에도 남은 critical {len(review['critical'])}건. 상세는 review.md 참조.\n\n"
        md = md.replace("## SUMMARY", warn + "## SUMMARY", 1)
    pdf, pages = render_pdf(md)
    paths = {"md": out / "report.md", "pdf": out / f"RAG-Output_{author['campus']}_{author['class']}_{author['name']}.pdf",
             "review": out / "review.md", "trace": out / "trace.json"}
    paths["md"].write_text(md)
    paths["pdf"].write_bytes(pdf)
    status = "approve (검토 통과)" if review["passed"] else "unverified (재작성 한도 도달)"
    paths["review"].write_text(
        f"# Review\n\n- 결과: {status}\n- 재작성 횟수: {state.get('revision_count', 0)}\n- PDF 쪽수: {pages}\n\n## critical\n"
        + "".join(f"- {c}\n" for c in review["critical"]) + "\n## minor\n" + "".join(f"- {c}\n" for c in review["minor"]))
    paths["trace"].write_text(_j({"trace": state["trace"], "sources": state["sources"]}))
    return {"outputs": {**{k: str(v.relative_to(ROOT)) for k, v in paths.items()}, "pages": pages, "review_passed": review["passed"]}}
