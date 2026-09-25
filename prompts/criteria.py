"""평가 설계 데이터: 관점별 기준(M1~M3, D1~D4)·등급 루브릭, 확증편향 방지 전략, 보고서 규칙.
README 의 평가 기준 표와 1:1 대응한다. 에이전트는 여기 정의만 보고 질의·프롬프트를 만든다."""

RATINGS = ["높음", "중간", "낮음", "판단 유보"]
CONFIDENCE = ["높음", "중간", "낮음"]

# evidence: paper = RAG(논문), web = 웹 검색, both = 둘 다
# rag_q: 논문 RAG 질의 / web_q: (채택·성과, 한계·제약) 사실 위주 질의 쌍 — criticism/review/opinion 질의 금지
CRITERIA = {
    "market": [
        {"id": "M1", "name": "시장 규모·성장성", "target": "기술이 속한 시장의 규모·전망", "evidence": "web",
         "high": "시장 리포트·업계 발표로 성장세 확인", "mid": "인접 시장 수치 등 간접 근거만", "low": "수요 불확실·축소 신호",
         "rag_q": None,
         "web_q": ("{market} market size growth forecast", "{market} market demand slowdown constraints data")},
        {"id": "M2", "name": "상용화·채택 현황", "target": "제품 출시, 서비스 도입, 오픈소스 통합", "evidence": "both",
         "high": "상용 제품·주요 서빙 스택 탑재 운영 중", "mid": "시제품·PoC·실험적 통합", "low": "논문·연구 단계",
         "rag_q": "Is {name} deployed in a product or service, or released as open-source code or checkpoints?",
         "web_q": ("{search} adoption production deployment product launch", "{search} deployment requirements compatibility constraints")},
        {"id": "M3", "name": "생태계 지지", "target": "지원 프레임워크, 표준화 동향(표준 규격·표준화 단체 채택)", "evidence": "both",
         "high": "복수 프레임워크 지원 또는 산업 표준 기반", "mid": "일부 커뮤니티 구현·표준 논의", "low": "독자 구현만 존재",
         "rag_q": "Which frameworks, standards, interconnects or software stacks does {name} build on or integrate with?",
         "web_q": ("{search} support vLLM SGLang TensorRT-LLM framework standard", "{search} limitations unsupported hardware software requirements")},
    ],
    "domain": [
        {"id": "D1", "name": "비용 효율", "target": "같은 워크로드에 필요한 GPU·메모리 비용 절감 여지", "evidence": "paper",
         "high": "추가 비용 없이 수용량 확대가 수치로 확인", "mid": "절감 효과 있으나 추가 투자·조건 필요", "low": "비용 증가 요인이 더 큼",
         "rag_q": "How much GPU memory, KV cache size or hardware cost does {name} save, and what extra cost does it require?",
         "web_q": None},
        {"id": "D2", "name": "처리량·지연", "target": "동시 요청 환경의 throughput·latency·SLO", "evidence": "paper",
         "high": "서빙 조건에서 처리량 향상 + 지연 유지", "mid": "특정 조건에서만 개선", "low": "지연·오버헤드 증가",
         "rag_q": "What throughput, latency or time-to-first-token results does {name} report, under which batch and context conditions?",
         "web_q": None},
        {"id": "D3", "name": "품질 유지", "target": "정확도·출력 품질 손실 위험", "evidence": "paper",
         "high": "정확도 벤치마크로 손실 없음/무시 가능", "mid": "조건부 손실", "low": "유의미한 손실",
         "rag_q": "What accuracy or benchmark quality results are reported for {name} compared with the baseline?",
         "web_q": None},
        {"id": "D4", "name": "도입 용이성", "target": "기존 인프라·서빙 스택 호환, 추가 HW, 운영 복잡도", "evidence": "both",
         "high": "SW 변경만으로 도입", "mid": "일부 커널·시스템 수정 또는 제한적 HW 추가", "low": "신규 인프라·장비 필요",
         "rag_q": "What hardware, system software, kernel or model changes are required to deploy {name}?",
         "web_q": ("{search} integration existing inference serving stack", "{search} deployment requirements hardware overhead")},
    ],
}

# 기술 조사 에이전트 질의 (논문 원문만 근거)
PROFILE_QUESTIONS = [
    "What problem does {name} address and what is its overall approach?",
    "How does the core mechanism of {name} work?",
    "What performance results do the authors report for {name}?",
    "What experimental setup, hardware, models and baselines are used to evaluate {name}?",
    "What limitations, overheads or trade-offs of {name} are stated?",
    "What is required to adopt or deploy {name} in practice?",
]

# 확증편향 방지 전략 7항목 — README·코드·보고서 6장에 동일하게 사용
BIAS_STRATEGIES = [
    "질의 쌍 구성: 기준마다 웹 검색을 '채택·성과' 질의와 '한계·제약(사실)' 질의 쌍으로 수행하고, 반응·의견 수집 질의(criticism/review/opinion)는 쓰지 않음",
    "반대 근거 강제: 모든 기준 평가에 counter_evidence 필드를 필수로 두어 지지 근거만 모으는 것을 구조적으로 차단",
    "근거 출처 구분: 논문 수치는 '저자 보고 기준'으로 표기하고 제3자(웹) 자료와 구분, 인접 기술·일반 시장 자료는 '간접 근거'로 표시",
    "판단 유보 등급: 근거가 부족하면 등급을 만들지 않고 '판단 유보'로 기록",
    "모델 분리: 생성(Generator)과 판정(Judge)에 서로 다른 LLM을 사용해 자기 평가 편향을 줄임",
    "이중 점검: 보고서 검토 단계에서 우열·추천 표현을 금칙어 규칙과 LLM Judge로 이중 검사",
    "Reflection: 조사·평가 결과를 인용한 논문 페이지 원문과 다시 대조(LLM 검증 + 수치 정규식 대조)",
]

# 보고서 규칙 (검토 에이전트의 결정론 검사)
REQUIRED_SECTIONS = ["SUMMARY", "분석 배경", "기술 선정", "기술 개요", "관점별 평가", "시사점", "한계점", "REFERENCE"]
FORBIDDEN_PHRASES = ["추천한다", "추천합니다", "추천된다", "권고한다", "더 우수", "우수하다", "우월", "승자", "열등", "우위에 있", "압도적"]
SUMMARY_MAX_CHARS = 900  # ponytail: 반 페이지 ≈ 900자 근사치, 폰트·여백 바꾸면 재보정
MAX_PAGES = 10


def criteria_table(perspective: str) -> str:
    rows = [f"| {c['id']} | {c['name']} | {c['target']} | {c['high']} | {c['mid']} | {c['low']} |" for c in CRITERIA[perspective]]
    return "| ID | 기준 | 평가 대상 | 높음 | 중간 | 낮음 |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
