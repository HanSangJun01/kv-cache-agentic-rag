"""RAG 컴포넌트 (에이전트 아님): 로딩 → 청킹 → 오픈소스 임베딩 → FAISS + BM25 하이브리드 검색(tech 필터)
→ Agentic 검색 루프(관련성 판정 · 질의 재작성) + 웹 검색 + 출처 레지스트리 + 수치-원문 대조."""
import hashlib
import html as html_lib
import os
import re
import threading
import urllib.parse
import urllib.request
from functools import cache
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS faiss+torch libomp 충돌 회피
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pymupdf
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from prompts.templates import GRADE_PROMPT, REWRITE_PROMPT

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, CACHE_DIR = ROOT / "data", ROOT / ".cache"

GENERATOR_MODEL = "gpt-4.1"
JUDGE_MODEL = "gpt-5.4-mini"  # Generator 와 다른 모델 (자기 평가 편향 회피)
EMBED_MODEL = "BAAI/bge-m3"  # rag/evaluate.py 비교 결과로 선정 (README 참조)
EMBED_QUERY_KWARGS = {  # 모델 카드 권장 질의 지시문
    "Qwen/Qwen3-Embedding-0.6B": {"prompt_name": "query"},
    "BAAI/bge-small-en-v1.5": {"prompt": "Represent this sentence for searching relevant passages: "},
}
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
TOP_K = 5
MAX_QUERY_REWRITE = 1
RRF_K = 60

PAPER_TAG = re.compile(r"\[(P-SW|P-HW) p\.(\d+)\]")


@cache
def generator():
    return ChatOpenAI(model=GENERATOR_MODEL, temperature=0)


@cache
def judge():
    return ChatOpenAI(model=JUDGE_MODEL, reasoning_effort="low")


# ---------------------------------------------------------------- 로딩 · 청킹 · 인덱스
def load_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [re.sub(r"-\n(?=[a-z])", "", p.get_text()) for p in doc]  # 줄끝 하이픈 복원


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def make_chunks(technologies: dict) -> tuple[dict, list[Document]]:
    """페이지 단위로 로딩 후 페이지 안에서 청킹 → 청크마다 인용 페이지가 정확함. 반환: (페이지 원문, 청크)"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pages, chunks = {}, []
    for tech, t in technologies.items():
        pages[t["source_id"]] = load_pages(DATA_DIR / t["file"])
        for page_no, text in enumerate(pages[t["source_id"]], 1):
            for piece in splitter.split_text(text):
                meta = {"tech": tech, "source_id": t["source_id"], "page": page_no, "idx": len(chunks)}
                chunks.append(Document(page_content=piece, metadata=meta))
    return pages, chunks


class KnowledgeBase:
    """기술별(tech) 논문 청크를 담는 인덱스. 페이지 원문도 보관해 인용 검증에 쓴다."""

    def __init__(self, technologies: dict, model: str = EMBED_MODEL):
        self.pages, self.chunks = make_chunks(technologies)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True, **EMBED_QUERY_KWARGS.get(model, {})})
        key = hashlib.md5(f"{model}{CHUNK_SIZE}{CHUNK_OVERLAP}{sorted((t['file'], tech) for tech, t in technologies.items())}".encode()).hexdigest()[:10]
        path = CACHE_DIR / f"faiss-{key}"
        if path.exists():  # 재실행 시 재임베딩 안 함
            self.vs = FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vs = FAISS.from_documents(self.chunks, self.embeddings)
            self.vs.save_local(str(path))
        self.bm25 = {}
        for tech in technologies:
            ids = [c.metadata["idx"] for c in self.chunks if c.metadata["tech"] == tech]
            self.bm25[tech] = (BM25Okapi([tokenize(self.chunks[i].page_content) for i in ids]), ids)
        self._lock = threading.Lock()

    def search(self, query: str, tech: str, k: int = TOP_K, mode: str = "hybrid") -> list[Document]:
        """tech 필터 필수. hybrid = Dense + BM25 를 RRF 로 병합."""
        with self._lock:
            ranked = []
            if mode in ("hybrid", "dense"):
                dense = self.vs.similarity_search(query, k=k * 2, filter={"tech": tech}, fetch_k=len(self.chunks))
                ranked.append([d.metadata["idx"] for d in dense])
            if mode in ("hybrid", "bm25"):
                bm25, ids = self.bm25[tech]
                scores = bm25.get_scores(tokenize(query))
                ranked.append([ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[: k * 2]])
        fused = {}
        for ranking in ranked:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0) + 1 / (RRF_K + rank + 1)
        return [self.chunks[i] for i in sorted(fused, key=lambda i: -fused[i])[:k]]

    def with_parent(self, doc: Document) -> str:
        """Parent Document 방식: 같은 페이지의 앞 청크를 문맥으로 붙인다."""
        i = doc.metadata["idx"]
        prev = self.chunks[i - 1] if i > 0 else None
        if prev and prev.metadata["source_id"] == doc.metadata["source_id"] and prev.metadata["page"] == doc.metadata["page"]:
            return prev.page_content[-400:] + "\n" + doc.page_content
        return doc.page_content

    def page_text(self, source_id: str, page: int) -> str | None:
        pages = self.pages.get(source_id, [])
        return pages[page - 1] if 0 < page <= len(pages) else None


_KB = {}


def knowledge_base(technologies: dict) -> KnowledgeBase:
    key = tuple(sorted((tech, t["file"]) for tech, t in technologies.items()))
    if key not in _KB:
        _KB[key] = KnowledgeBase(technologies)
    return _KB[key]


def tag(doc: Document) -> str:
    return f"[{doc.metadata['source_id']} p.{doc.metadata['page']}]"


# ---------------------------------------------------------------- Agentic 검색 루프
class Grades(BaseModel):
    relevant_ids: list[int] = Field(description="질문에 답하는 데 실제로 쓸 수 있는 문서 번호")


def agentic_retrieve(kb: KnowledgeBase, question: str, tech: str, name: str) -> tuple[list[dict], dict]:
    """검색 → 관련성 판정(Judge) → 0건이면 질의 재작성 후 재검색(최대 MAX_QUERY_REWRITE회).
    반환: (근거 리스트 [{tag, text}], trace 항목)"""
    trace = {"agent_tech": tech, "question": question, "attempts": []}
    query, relevant = question, []
    for attempt in range(MAX_QUERY_REWRITE + 1):
        docs = kb.search(query, tech)
        listing = "\n\n".join(f"[{i}] {tag(d)}\n{d.page_content[:1200]}" for i, d in enumerate(docs))
        grades = judge().with_structured_output(Grades).invoke(GRADE_PROMPT.format(name=name, question=question, docs=listing))
        relevant = [d for i, d in enumerate(docs) if i in set(grades.relevant_ids)]
        trace["attempts"].append({"query": query, "retrieved": [tag(d) for d in docs], "relevant": [tag(d) for d in relevant]})
        if relevant or attempt == MAX_QUERY_REWRITE:
            break
        query = generator().invoke(REWRITE_PROMPT.format(name=name, question=question)).content.strip()
    trace["result"] = f"{len(relevant)}건 채택" if relevant else "관련 근거 없음"
    return [{"tag": tag(d), "text": kb.with_parent(d)} for d in relevant], trace


# ---------------------------------------------------------------- 웹 검색 · 출처 레지스트리
def web_search(query: str, max_results: int = 3) -> list[dict]:
    from tavily import TavilyClient
    try:
        return TavilyClient().search(query, max_results=max_results).get("results", [])
    except Exception as e:  # 웹 실패가 보고서 생성 자체를 막지 않도록
        print(f"  ! web search failed: {query} ({e})")
        return []


@cache
def page_meta(url: str) -> dict[str, list[str]]:
    """페이지 메타데이터 — REFERENCE 형식(논문/특허/웹) 판별과 게시일·제목 보강(n.d.·잘린 제목 방지)에 쓴다. 실패 시 {}."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/126 Safari/537.36", "Accept": "text/html,*/*"})
        return parse_meta(urllib.request.urlopen(req, timeout=8).read(400_000).decode("utf-8", "ignore"))
    except Exception:
        return {}


def parse_meta(page: str) -> dict[str, list[str]]:
    """<meta> → {name: [content, ...]}. scheme 이 있으면 'name.scheme' 키 (예: dc.contributor.assignee)."""
    meta = {}
    for tag_ in re.findall(r"<meta\s[^>]*>", page, re.I):
        attrs = {k.lower(): html_lib.unescape(a or b or c.rstrip("/"))  # 따옴표 없는 속성(name=citation_title)도 허용
                 for k, a, b, c in re.findall(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", tag_)}
        name = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        if name and attrs.get("content", "").strip():
            key = f"{name}.{attrs['scheme'].lower()}" if attrs.get("scheme") else name
            meta.setdefault(key, []).append(attrs["content"].strip())
    if m := re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S):
        meta["_title"] = [html_lib.unescape(" ".join(m.group(1).split()))]
    if m := re.search(r"""\"datePublished\"\s*:\s*\"([^"]+)\"|<time[^>]*datetime=["']([^"']+)""", page):
        meta["_date"] = [m.group(1) or m.group(2)]
    return meta


def norm_date(text: str) -> str:
    """'2025/7/3', '2025-07', '2025' → 가능한 만큼 YYYY-MM-DD (없으면 '')."""
    m = re.search(r"(20\d{2})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?", text or "")
    return "-".join(f"{int(g):02d}" if i else g for i, g in enumerate(m.groups()) if g) if m else ""


def author_short(names: list[str]) -> str:
    """['Geens, Robin', ...] 또는 ['Tao Ji', ...] → 'Geens, R. et al.'"""
    if not names:
        return ""
    first = re.sub(r"\s*\(.*?\)", "", names[0]).strip()
    if "," in first:
        last, given = [x.strip() for x in first.split(",", 1)]
    else:
        *given_parts, last = first.split()
        given = " ".join(given_parts)
    initials = " ".join(f"{g[0]}." for g in re.split(r"[\s-]+", given) if g)
    return (f"{last}, {initials}" if initials else last) + (" et al." if len(names) > 1 else "")


def format_citation(result: dict, meta: dict, url: str) -> tuple[str, str, str, str, str]:
    """REFERENCE 표기 지침에 맞춘 서지 문자열. 반환: (citation, title, site, date, ref_type=논문/특허/기타)
    - 특허: 출원인(YYYY-MM). *특허명*, 특허번호/공개번호, URL
    - 논문: 저자(YYYY). 논문제목. *학술지/학회명*, 권(호), 페이지.
    - 웹:   기관명 또는 작성자(YYYY-MM-DD). *제목*. 사이트명, URL"""
    g = lambda k: (meta.get(k) or [""])[0]
    em = lambda t: t.replace("*", r"\*")  # 제목 안의 * 가 기울임 표기를 깨지 않도록
    host = re.sub(r"^www\.", "", urllib.parse.urlparse(url).netloc)
    site = g("og:site_name") or host
    title = (result.get("title") or "").strip()
    if not title or re.search(r"(\.\.\.|…)\s*$", title):  # 검색 API 가 자른 제목 → 페이지 제목으로 복원
        full = g("citation_title") or g("dc.title") or g("og:title") or g("_title")
        title = full if full and len(full) <= 150 else title  # 설명문까지 붙은 긴 페이지 제목(GitHub 등)은 쓰지 않음
    for sep in (" | ", " – ", " - "):  # "제목 | 사이트명" 의 사이트명 꼬리 제거
        if title.lower().endswith((sep + site).lower()):
            title = title[: -len(sep + site)]
    title = re.sub(r"\s*(\.\.\.+|…)\s*$", "", title).strip().rstrip(" &,:;|–-")  # 복원 못 한 잘린 제목의 꼬리 정리

    if g("citation_patent_publication_number") or "patents.google" in host:
        number = g("citation_patent_publication_number").replace(":", "-")
        if not number and (m := re.search(r"/patent/([A-Z]{2}\d+[A-Z]\d?)", url)):
            number = m.group(1)
        date = norm_date(g("dc.date"))[:7] or "n.d."
        applicant = g("dc.contributor.assignee") or site
        title = g("dc.title") or title
        return f"{applicant}({date}). *{em(title)}*, {number}, {url}", title, site, date, "특허"
    if g("citation_title"):
        title = g("citation_title")
        year = norm_date(g("citation_publication_date") or g("citation_date") or g("citation_online_date"))[:4] or "n.d."
        venue = g("citation_conference_title") or g("citation_journal_title") or ("arXiv" if g("citation_arxiv_id") else site)
        extra = ""
        if g("citation_volume"):
            extra += f", {g('citation_volume')}" + (f"({g('citation_issue')})" if g("citation_issue") else "")
        if g("citation_firstpage"):
            extra += f", {g('citation_firstpage')}" + (f"-{g('citation_lastpage')}" if g("citation_lastpage") else "")
        if g("citation_arxiv_id") and venue == "arXiv":
            extra += f", {g('citation_arxiv_id')}"
        return f"{author_short(meta.get('citation_author', [])) or site}({year}). {title}. *{venue}*{extra}.", title, site, year, "논문"

    url_date = re.search(r"/(20\d{2})/(\d{2})(?:/(\d{2}))?/", url)  # 블로그·뉴스 URL 의 /YYYY/MM/DD/
    date = (norm_date(result.get("published_date", ""))
            or norm_date(next((g(k) for k in ("article:published_time", "datepublished", "publish_date", "pubdate", "date", "dc.date", "_date") if g(k)), ""))
            or (norm_date("-".join(x for x in url_date.groups() if x)) if url_date else "") or "n.d.")
    author = g("author")
    org = author if author and not author.startswith("http") else site
    return f"{org}({date}). *{em(title)}*. {site}, {url}", title, site, date, "기타"


def landing_url(url: str) -> str:
    """논문 PDF 링크 → 서지 메타데이터가 있는 초록 페이지 (arXiv, ACL Anthology)."""
    url = re.sub(r"arxiv\.org/pdf/([\d.]+?)(v\d+)?(\.pdf)?$", r"arxiv.org/abs/\1", url)
    return re.sub(r"(aclanthology\.org/[\w.-]+?)\.pdf$", r"\1", url)


def web_source(result: dict, source_id: str, tech: str) -> dict:
    url = result["url"]
    citation, title, site, date, ref_type = format_citation(result, page_meta(landing_url(url)), url)
    return {"id": source_id, "kind": "web", "tech": tech, "title": title, "url": url, "site": site, "date": date,
            "ref_type": ref_type, "citation": citation, "text": result.get("content", "")}


# ---------------------------------------------------------------- 수치-원문 대조 (결정론, LLM 없음)
NUM_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def key_numbers(text: str) -> list[str]:
    """검증 대상 수치: 소수 또는 10 이상 정수 (연도·장 번호 같은 작은 정수 제외)."""
    out = []
    for n in NUM_RE.findall(text):
        v = float(n.replace(",", ""))
        if "." in n or (v >= 10 and not 1900 <= v <= 2100):
            out.append(n)
    return out


def number_in(num: str, page: str) -> bool:
    n = num.replace(",", "")
    variants = {n, n.rstrip("0").rstrip(".")} if "." in n else {n}
    page = page.replace(",", "")
    return any(re.search(rf"(?<![\d.]){re.escape(v)}(?!\d)", page) for v in variants)


def unsupported_numbers(text: str, cite_re: re.Pattern, page_of) -> list[str]:
    """인용 태그 바로 앞 구간의 수치가 인용 페이지 원문에 있는지 대조. page_of(match) → 페이지 원문 | None(웹)."""
    group_re = re.compile(rf"(?:\s*(?:{cite_re.pattern}))+")
    issues = []
    for sent in re.split(r"(?<=[.!?。])\s+|\n+", text):
        last = 0
        for g in group_re.finditer(sent):
            seg, last = sent[last:g.start()], g.end()
            pages = [p for p in (page_of(m) for m in cite_re.finditer(g.group())) if p]
            seg_nums = key_numbers(re.sub(r"\[[^\]]*\]", " ", seg))  # 다른 인용 번호([11] 등)는 수치가 아님
            missing = [n for n in seg_nums if pages and not any(number_in(n, p) for p in pages)]
            if missing:
                issues.append(f"수치 {missing} 가 인용 페이지 {g.group().strip()} 원문에 없음: \"…{seg.strip()[-90:]}\"")
    return issues

