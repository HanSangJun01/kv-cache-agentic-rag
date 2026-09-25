"""임베딩 후보 비교 + 최종 검색기 품질 측정 (Hit Rate@K, MRR). 실행: python -m rag.evaluate
평가셋: 청크 → LLM 이 에이전트식 질문 생성 → (질문, 정답 청크). 절반은 모델 선정용, 나머지 절반(held-out)은 최종 수치용."""
import json
import random
import time

from pydantic import BaseModel

from app import TECHNOLOGIES
from rag.pipeline import ROOT, KnowledgeBase, generator, make_chunks

CANDIDATES = {  # 선정 기준: 영문 기술 문서 적합성, 최대 시퀀스 ≥ 청크 길이, 차원·메모리, 로컬 CPU 추론 비용, 상업 이용 가능 라이선스
    "BAAI/bge-m3": "568M · 1024d · 8192 tok · MIT",
    "Qwen/Qwen3-Embedding-0.6B": "596M · 1024d · 32k tok · Apache-2.0",
    "BAAI/bge-small-en-v1.5": "33M · 384d · 512 tok · MIT",
}
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
    for i, c in enumerate(pool[:N_QUESTIONS]):
        name = TECHNOLOGIES[c.metadata["tech"]]["name"]
        q = generator().with_structured_output(Q).invoke(
            f"Write one short English question that a technology analyst evaluating {name} would ask "
            f"(about its method, results, cost, limitations or deployment) and that this passage specifically answers. "
            f"Paraphrase; do not copy phrases or mention 'the passage'.\n\nPassage:\n{c.page_content}").question
        items.append({"question": q, "tech": c.metadata["tech"], "gold": c.metadata["idx"],
                      "split": "selection" if i % 2 == 0 else "test"})
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
    sel, test = [i for i in items if i["split"] == "selection"], [i for i in items if i["split"] == "test"]
    result = {"n_selection": len(sel), "n_test": len(test), "candidates": {}}
    kbs = {}
    for model, spec in CANDIDATES.items():
        t = time.time()
        kbs[model] = KnowledgeBase(TECHNOLOGIES, model=model)
        result["candidates"][model] = {"spec": spec, "index_sec(cache 포함)": round(time.time() - t, 1), **score(kbs[model], sel, "dense")}
        print(model, result["candidates"][model])
    chosen = max(result["candidates"], key=lambda m: result["candidates"][m]["mrr@10"])
    result["chosen"] = chosen
    result["held_out"] = {mode: score(kbs[chosen], test, mode) for mode in ("dense", "bm25", "hybrid")}
    print("chosen:", chosen, result["held_out"])
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
