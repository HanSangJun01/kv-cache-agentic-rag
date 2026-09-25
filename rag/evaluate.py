"""검색 품질 측정 (Hit Rate@K, MRR) — 고정 임베딩(bge-m3)으로 Dense / BM25 / Hybrid(사용 중) 비교. 실행: python -m rag.evaluate
평가셋: 청크 → LLM 이 분석가식 질문 생성 → (질문, 정답 청크). outputs/retrieval_eval_set.json 에 저장해 재사용 (있으면 LLM 호출 없음)."""
import json
import random

from pydantic import BaseModel

from app import TECHNOLOGIES
from rag.pipeline import EMBED_MODEL, ROOT, KnowledgeBase, generator, knowledge_base, make_chunks

N_QUESTIONS, K_LIST = 60, (1, 3, 5)
SET_PATH, OUT_PATH = ROOT / "outputs/retrieval_eval_set.json", ROOT / "outputs/retrieval_eval.json"


class Q(BaseModel):
    question: str


def build_eval_set() -> list[dict]:
    if SET_PATH.exists():
        return json.loads(SET_PATH.read_text())
    _, chunks = make_chunks(TECHNOLOGIES)
    pool = [c for c in chunks if len(c.page_content) > 500 and c.page_content.count("et al.") < 3]  # 참고문헌 청크 제외
    random.Random(42).shuffle(pool)
    items = []
    for c in pool[:N_QUESTIONS]:
        name = TECHNOLOGIES[c.metadata["tech"]]["name"]
        q = generator().with_structured_output(Q).invoke(
            f"Write one short English question that a technology analyst evaluating {name} would ask "
            f"(about its method, results, cost, limitations or deployment) and that this passage specifically answers. "
            f"Paraphrase; do not copy phrases or mention 'the passage'.\n\nPassage:\n{c.page_content}").question
        items.append({"question": q, "tech": c.metadata["tech"], "gold": c.metadata["idx"]})
    SET_PATH.parent.mkdir(exist_ok=True)
    SET_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    return items


def score(kb: KnowledgeBase, items: list[dict], mode: str) -> dict:
    hits, rr = {k: 0 for k in K_LIST}, 0.0
    for it in items:
        ranked = [d.metadata["idx"] for d in kb.search(it["question"], it["tech"], k=10, mode=mode)]
        rank = ranked.index(it["gold"]) + 1 if it["gold"] in ranked else None
        rr += 1 / rank if rank else 0
        for k in K_LIST:
            hits[k] += bool(rank and rank <= k)
    n = len(items)
    return {**{f"hit@{k}": round(hits[k] / n, 3) for k in K_LIST}, "mrr@10": round(rr / n, 3)}


def main():
    items = build_eval_set()
    kb = knowledge_base(TECHNOLOGIES)
    result = {"embedding": EMBED_MODEL, "n_questions": len(items),
              **{mode: score(kb, items, mode) for mode in ("dense", "bm25", "hybrid")}}
    print(result)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
