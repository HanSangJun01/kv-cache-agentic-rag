# RAG 강의·노트북 참고 메모 (과제용)

> 출처: 강의 PDF `6. RAG Pipeline 설계 및 구축`(배기주, 164p) + `~/langgraph-v1/20-RAG/` 노트북 9개, `rag/*.py`
> 참고용. 설계 결정은 자유. 필요할 때 해당 페이지/노트북만 찾아보면 됨.

---

## 1. 파이프라인 기본 (p.26~28)
Loader → Splitter → Embedding → VectorDB → Retriever → Prompt → LLM → Chain

- 노트북 기본값: `PDFPlumberLoader`, `RecursiveCharacterTextSplitter(1000/100 또는 1200/200)`, FAISS / InMemoryVectorStore
- overlap은 chunk의 10~20% (p.32)
- 전처리 팁 (p.41): 페이지 단위 baseline → 표는 캡션과 묶기, header/footer 제거, 하이픈 복원
- 메타데이터에 source/page 넣어두면 출처 표기·REFERENCE 생성이 편함 (p.40, `rag/utils.py format_docs`)

## 2. Embedding 선택 (p.47~49)
- 고려 축: 언어 / 도메인 / 문서 길이 / 차원 / 비용
- p.49: *"리더보드는 검토 시작점, 최종은 데이터셋에 적용해서 선정"* — 채점 기준("리더보드 상위 ← 부적절")과 같은 말
- 강의 MTEB 표의 오픈소스 후보 예: bge-m3 (1024d), Qwen3-Embedding-0.6B (1024d), embeddinggemma-300m (768d), jina-v4, Qwen3-4B/8B
- 일반 Q&A RAG 적정 차원 768~1024 (p.56)

## 3. Retriever & 평가 (p.65~89)
- Cosine(기본) / MMR(여러 관점 모을 때, p.67) / Hybrid BM25+Dense, RRF 병합(p.70) / Reranker(p.72~78)
- 강의 예시(p.89)에선 Ensemble(BM25+FAISS)이 MRR 최고
- **Hit Rate@K** = 상위 K 안에 정답 있는 비율, **MRR** = mean(1/rank) (p.86)
- 평가셋 만들기 (p.88): 청크 → LLM에게 질문 생성 → (질문, chunk_id)가 Ground Truth. 라벨링 불필요

## 4. 노트북에서 가져다 쓸 수 있는 패턴
| 노트북 | 패턴 |
|---|---|
| 01 Naive | GraphState / retrieve → answer 기본 골격, `format_docs` |
| 02 Relevance | 관련성 yes/no 판정 + `retrieve_count` 루프 상한 + fallback |
| 03 WebSearch | Tavily 웹검색 노드 (RAG 문서 세트 밖 정보 보강) |
| 04 QueryRewrite | 질문 재작성 프롬프트, `rewrite_count` |
| 10 Self-RAG | 문서별 grade 필터, Groundedness(환각) / GradeAnswer 구조화 출력 |
| 11 CRAG | correct / ambiguous / incorrect 3단계 판정, knowledge strip 정제, 웹 결과 URL을 `<source>`로 보존 |
| 12 Adaptive | `RouteQuery` 구조화 라우터 (vectorstore vs web) |
| 13 Agentic | `@tool` retriever + `ToolNode` + `tools_condition`, `recursion_limit` + `GraphRecursionError` |
| 14 advanced | 재시도 카운터 상한(`N_SQL_RETRY` 등), `ground_check`(답변 숫자를 출처와 코드로 대조), `INIT` state, RAGAS 평가 코드 |
| `rag/base.py` | 추상 체인 + `CacheBackedEmbeddings` + FAISS 인덱스 해시 캐시 (재실행 시 재임베딩 안 함) |

## 5. LangGraph 규칙 (p.141~147)
- State: TypedDict. Reducer 없으면 덮어쓰기, 있으면 누적(`operator.add`, `add_messages`)
- 병렬(fan-out) 시 에이전트별 결과는 **분리된 키**로 — 가이드라인 D절과 동일
- Node는 변경된 키만 dict로 반환. 한 노드에 여러 책임 몰지 않기
- 반복 흐름은 exit 조건 명확히 + `recursion_limit`
- Modular 패턴 용어 (p.117~121): Linear / Routing / Branching / Loop

## 6. 중립성·평가 관련 (p.132, p.156~157)
- Agentic RAG 한계: self-evaluation은 있어도 fact-check 내장 아님 → 결정론적 검증 병행 가능
- LLM-as-a-Judge 주의: 일관성 부족, 관대함, **자기 모델 평가 문제** → Judge와 Generator 모델 분리 고려
- Judge 프롬프트는 기준 명시 + form-filling (p.87 G-Eval 예시)
- RAGAS: Faithfulness, ResponseRelevancy, ContextPrecision(WithoutReference) — 정답 없이도 계산 가능 (14번 코드)

## 7. 노트북 코드 이식 시 주의
- `OpenAIEmbeddings` → 오픈소스(예: `HuggingFaceEmbeddings`)로 교체 필요
- `langchain_teddynote` 의존 (logging, GroundednessChecker, TavilySearch, visualize_graph, stream_graph) — 재현성 위해 표준 API로 대체하거나 requirements에 명시
- `hub.pull("teddynote/rag-prompt")` → 로컬 프롬프트 파일로
- macOS FAISS+torch libomp 충돌 시 `KMP_DUPLICATE_LIB_OK=TRUE` (14번 첫 셀)

## 8. 배경 한 줄 (p.11, p.131)
Naive → Advanced → Self-RAG → CRAG → Adaptive → Modular → Agentic(2025.01).
Agentic RAG 적용 사례 = *"In-depth Q&A, 검색 기반 보고서 생성"* — 과제가 이 유형.

## 9. 안 쓴 것
Text-to-SQL, LLM Wiki 논쟁, 법령/Graph RAG 파싱, Late/Semantic chunking, SPLADE, IVF/HNSW/PQ 상세, ROUGE/BLEU/METEOR, LangSmith UI, Chain of Density
