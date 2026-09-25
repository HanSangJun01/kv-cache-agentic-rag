# KV cache 최적화 기술 다관점 평가 보고서

**SW 압축 vs HW 메모리 확장** — DeepSeek-V2 MLA · ITME (CXL-Hybrid 계층 메모리 확장)  
평가 관점: 시장성 · 도메인 적용(데이터센터·클라우드 LLM 서빙) | 작성일: 2026-09-25 | 자동 생성(LangGraph Multi-Agent + Agentic RAG)

## SUMMARY

- SW(DeepSeek-V2 MLA)와 HW(ITME, CXL-hybrid 계층 메모리 확장) 모두 LLM 서빙에서 처리량 및 비용 효율성 개선에 기여함을 시장성·도메인 관점 모두 인정한다([1, p.1][1, p.21][2, p.1][2, p.11]).
- 시장성 관점에서는 SW는 주요 오픈소스 프레임워크(vLLM, SGLang, TensorRT-LLM 등) 공식 지원 및 DeepSeek 계열 실서비스 배포 근거로 도입 장벽이 낮게 인식되나, 도메인 관점에서는 구조 변경 필요성으로 도입 난이도가 더 높게 평가된다([3][4][5][6][7][1, p.6][1, p.29]).
- HW는 시장성 관점에서 CXL 메모리 확장 시장의 성장과 일부 상용화 사례로 높은 성장성을 인정받으나, 도메인 관점에서는 추가 하드웨어 투자·운영 복잡도 등 실도입 장벽이 강조된다([8][9][10][11][2, p.1][2, p.6][2, p.11]).
- 품질·정확도 영향에 대해 시장성 관점은 별도 우려를 제기하지 않으나, 도메인 관점에서는 SW는 MLA 적용 전후 품질 비교 수치, HW는 정확도 벤치마크 부재로 한계를 지적한다([1, p.21][2, p.11]).
- SW는 소프트웨어 구조 변경, HW는 하드웨어 인프라 확장에 초점을 두며, 두 방식은 병행 적용 가능성이 있으나 결합 효과는 검증되지 않았다([1, p.8][2, p.2]).

---

## 1. 분석 배경

KV cache는 LLM 추론에서 연산 병목을 메모리(HBM 용량) 병목으로 전환시키는 핵심 요인으로 부상했다. 기존에는 연산량이 주요 제약이었으나, 대규모 컨텍스트와 동시 요청이 증가하면서 KV cache가 GPU 메모리의 주요 점유원이 되어, 처리량·비용 효율성의 한계로 작용한다. 이에 따라, SW(데이터를 작게)와 HW(공간을 넓게) 두 진영이 각각 어텐션 구조 변경 또는 메모리 계층 확장 방식으로 병목 해소를 시도하고 있다. 이 두 방식은 상호 대체가 아니라 병행 적용도 가능하다.

본 보고서는 데이터센터·클라우드 LLM 서빙 도메인에서, 시장성(시장 규모·채택·생태계)과 도메인 적용(비용 효율·처리량·품질·도입 용이성) 두 관점만을 평가 범위로 삼는다. 기술성숙도, 이해관계자(경쟁사·개발자·투자자) 관점은 제외한다.

---

## 2. 기술 선정

본 평가는 에이전트 기반 선정 대신 Human 기반 방식으로, 과제 Doc Pool 에서 SW·HW 진영별 1개 기술을 직접 선정했다.

- SW: DeepSeek-V2 MLA(어텐션 아키텍처 개선)는 KV cache를 사후 압축하지 않고 어텐션 구조(Multi-head Latent Attention) 자체를 바꿔 KV cache 93.3% 감소를 보고한 기술이다. 공개 모델·오픈소스 서빙 스택 채택 사례가 풍부하고, 재학습이 필요한 구조 변경이라 도메인 도입 관점과 대비가 뚜렷하다.
- HW: ITME(CXL-Hybrid 계층 메모리 확장)는 원문 KV를 그대로 두고 CXL-Hybrid 메모리로 TB급 바이트 주소 공간을 확장하는 방식이다. 데이터센터 공유 컨텍스트 인프라를 직접 겨냥하며, 2026년 발표된 메모리 기업(SK hynix) 연구로 시장 근거가 간접적이라는 점 자체가 관점 간 차이를 드러낸다.

---

## 3. 기술 개요

### DeepSeek-V2 MLA (SW · 어텐션 아키텍처 개선)

- **핵심 방식**: DeepSeek-V2 MLA는 기존 Transformer의 Multi-Head Attention(MHA) 대비, KV cache를 잠재 벡터로 압축하는 저차원 결합 압축(low-rank key-value joint compression) 방식을 도입한다. 각 어텐션 헤드별로 decoupled query/key를 생성하고, RoPE를 적용한 후, KV cache를 (dc+dh^R)l 크기로 유지한다. GQA(Grouped-Query Attention) 2.25 그룹 수준의 KV 캐시만 필요로 하면서도 MHA보다 더 나은 성능을 달성한다(저자 보고 기준)[1, p.6][1, p.8].
- **저자 보고 성과**: DeepSeek-V2는 DeepSeek 67B 대비 학습 비용 42.5% 절감, KV 캐시 93.3% 절감, 최대 생성 처리량 5.76배 향상(저자 보고 기준)[1, p.1][1, p.21]. 21B 활성화 파라미터로 오픈소스 모델 중 최고 수준의 성능 및 오픈소스 MoE 모델 중 최강 성능을 달성했다고 보고한다[1, p.21].
- **한계**: 프리트레이닝 이후 지속적 지식 업데이트 불가, 비사실적 정보 및 환각(hallucination) 생성 가능성 존재, 주로 중국어와 영어 데이터로 학습되어 기타 언어에서는 성능이 제한될 수 있음(저자 보고 기준)[1, p.21].
- **실험 환경**: 60개 Transformer 레이어, hidden dimension 5120, MLA 어텐션 헤드 128개, 헤드당 차원 128, KV 압축 차원 512, 쿼리 압축 차원 1536, decoupled query/key 헤드당 차원 64. FFN은 MoE로 대체, 프리트레이닝 코퍼스 8.1T 토큰(중국어가 영어보다 약 12% 많음)[1, p.12].
- **도입 요건**: MLA 적용을 위해 KV 캐시 구조가 (dc+dh^R)l 형태로 구현되어야 함[1, p.8].

### ITME (HW · CXL-Hybrid 계층 메모리 확장)

- **핵심 방식**: ITME는 CXL-hybrid memory(내부 DRAM 캐시+SSD)를 활용, TB급의 byte-addressable 원격 메모리 확장을 제공한다. LLM 추론 워크로드의 결정적 데이터 접근 패턴을 활용한 pipelined, multi-tier DMA 기반 prefetching으로, GPU가 RDMA를 통해 대용량 모델 가중치와 KV 캐시에 직접 접근한다. 소프트웨어·하드웨어 prefetcher가 결합되어 PCIe 대역폭에 근접한 데이터 전송을 실현한다[2, p.2][2, p.3][2, p.7][2, p.8].
- **저자 보고 성과**: ITME는 conventional CPU-offloading 대비 최대 35.7% throughput 개선, Llama-3.1 8B/70B 모델 256 concurrent conversation 환경에서 recomputation-only baseline 대비 turn 5에서 최대 1.81× speedup(ideal GPU memory 80GB 대비 3.02×)을 보였다. host memory 사용량은 staging buffer(예: 30GB)로 제한하였다(저자 보고 기준)[2, p.1][2, p.9][2, p.11].
- **한계**: remote CXL-hybrid memory의 latency에 의해 local GPU memory 대비 성능이 제한되며, staging buffer 크기와 DRAM cache 용량에 따라 비용-성능 trade-off가 존재한다. LLM 워크로드의 결정적 접근 패턴에 의존한다[2, p.2][2, p.3][2, p.6][2, p.9][2, p.10].
- **실험 환경**: GPU 서버(T1/T2)와 remote CXL-hybrid memory 서버(T3.5), SK Hynix CMM, PCIe Gen5 NVMe SSD, FPGA 기반 하드웨어 프로토타입, vLLM(v0.17.0) 기반 실험, Llama-3.1 8B/70B, 256 concurrent conversation, 35-turn, ShareGPT 데이터셋 등[2, p.8][2, p.9][2, p.10][2, p.11].
- **도입 요건**: CXL-hybrid memory(내부 DRAM cache+SSD), PCIe Gen5 NVMe SSD, SK Hynix CMM 등 호환 하드웨어, RDMA 네트워크, 호스트 서버에 staging buffer(메모리) 할당 필요[2, p.1][2, p.2][2, p.6][2, p.8][2, p.11].

---

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| M1 시장 규모·성장성 | 높음 (신뢰도 중간) | 높음 (신뢰도 중간) |
| M2 상용화·채택 현황 | 높음 (신뢰도 높음) | 중간 (신뢰도 중간) |
| M3 생태계 지지 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |

#### SW: DeepSeek-V2 MLA

- **M1(시장 규모·성장성)**: 글로벌 LLM 추론 엔진 및 LLM 시장, 엔터프라이즈 LLM 시장 모두 2025~2034년 연평균 21~26% 성장률과 수십~수백억 달러 규모로 전망된다. 이는 MLA 등 특정 아키텍처가 아닌 시장 전체 성장세에 대한 간접 근거임을 명확히 한다. MLA 자체의 시장 점유율·직접적 성장 수치는 없다[12][13][14].
- **M2(상용화·채택 현황)**: DeepSeek-V2 MLA는 실제 서비스에 배포되어 8개 H800 GPU 노드에서 5.76배의 처리량 향상(저자 보고 기준) 등 실측 결과가 보고되었고, DeepSeek-V2, V3 등 주요 모델의 프로덕션 서빙 아키텍처로 사용된다. vLLM, SGLang, TensorRT-LLM 등 주요 오픈소스 LLM 서빙 프레임워크에서 MLA 최적화 커널이 공식 지원된다. 다만, MLA 적용 모델의 배포·운영은 DeepSeek 계열 및 일부 파생 모델에 집중되어 있다[3][4][5][6][7][1, p.16].
- **M3(생태계 지지)**: vLLM, SGLang, TensorRT-LLM 등 복수 프레임워크에서 MLA 공식 지원, DeepSeek 계열 모델에서의 채택은 명확하나, broader 생태계 채택은 블로그성 언급에 그치며, 표준화 단체의 공식 규격 채택 근거는 없다.

#### HW: ITME (CXL-Hybrid 계층 메모리 확장)

- **M1(시장 규모·성장성)**: CXL-hybrid memory 및 메모리 확장 시장은 2025~2034년 연평균 29% 이상의 성장률과 수십억 달러 규모로 전망된다. AI/LLM 추론, 데이터센터, HPC 등에서 대용량·저지연 메모리 수요가 CXL 시장 성장의 주요 동인으로 언급된다. ITME 자체의 시장 규모가 아닌 CXL 메모리 확장 시장 전체 수치이므로 간접 근거임을 명시한다[8][9][15].
- **M2(상용화·채택 현황)**: CXL-hybrid memory 기반 대용량 메모리 확장 및 KV cache 오프로딩은 Penguin Solutions, Astera Labs 등 실제 상용 제품과 일부 클라우드 서비스에서 도입되고 있다. 다만, ITME와 동일한 계층형(SSD+DRAM) 구조 및 LLM 추론 특화 아키텍처가 대규모로 운영 중임을 입증하는 독립적 근거는 부족하다[10][11][2, p.2][2, p.11].
- **M3(생태계 지지)**: CXL 메모리 확장 기술은 vLLM 등 일부 LLM 서빙 프레임워크에서 지원 논의 및 개발이 진행 중이며, Astera Labs, Penguin Solutions 등은 기존 AI 서빙 스택과 호환되는 CXL 메모리 솔루션을 제공한다. ITME의 계층형 하이브리드 구조 및 prefetching 방식이 산업 표준으로 채택되었다는 근거는 부족하다[16][17][18][2, p.2][2, p.11].

#### 비교·대조

- SW는 실제 서비스 배포 및 오픈소스 프레임워크 공식 지원 등 상용화·생태계 측면에서 일정 수준의 시장성을 보이나, DeepSeek 계열 외 확산 및 표준화 근거는 부족하다.
- HW는 시장 성장성과 일부 상용화 사례가 있으나, ITME 고유 구조의 대규모 상용화·표준화 근거는 제한적이다.
- 두 기술 모두 시장성 평가의 수치는 기술 전체가 아닌 시장 전체 성장세에 대한 간접 근거임을 유의해야 한다.

---

### 4.2 도메인 적용 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| D1 비용 효율 | 높음 (신뢰도 높음) | 중간 (신뢰도 중간) |
| D2 처리량·지연 | 중간 (신뢰도 높음) | 중간 (신뢰도 중간) |
| D3 품질 유지 | 중간 (신뢰도 높음) | 판단 유보 (신뢰도 낮음) |
| D4 도입 용이성 | 중간 (신뢰도 중간) | 낮음 (신뢰도 중간) |

#### SW: DeepSeek-V2 MLA

- **D1(비용 효율)**: DeepSeek-V2는 DeepSeek 67B 대비 KV 캐시 93.3% 절감, 최대 생성 처리량 5.76배 향상(저자 보고 기준) 등 수치적 개선이 확인된다[1, p.1][1, p.21][1, p.16].
- **D2(처리량·지연)**: 실제 서빙 환경에서 DeepSeek 67B 대비 5.76배의 최대 생성 처리량을 달성(저자 보고 기준). 대규모 동시 요청 효율성이나 지연 변화에 대한 직접적 수치는 없다[1, p.16][1, p.21].
- **D3(품질 유지)**: 정확도 벤치마크(MMLU, CHID, CCPM 등)에서 최고 수준의 성능을 보인다고 보고되나, MLA 적용 전후의 품질 손실 여부에 대한 직접적 비교 수치는 없다[1, p.15][1, p.21].
- **D4(도입 용이성)**: MLA는 Transformer 내 attention 모듈에서 KV 캐시를 저차원 잠재 벡터로 압축하는 구조적 변경이 필요하다(저자 보고 기준). 완전한 drop-in replacement임을 뒷받침하는 직접적 논문 근거는 없다[1, p.6][1, p.29][19][20][21].

#### HW: ITME (CXL-Hybrid 계층 메모리 확장)

- **D1(비용 효율)**: ITME는 conventional CPU-offloading 대비 최대 35.7% throughput 개선, host memory 사용량을 staging buffer(예: 30GB)로 제한하여 대규모 모델의 메모리 비용 부담을 줄일 수 있다. 그러나 CXL-hybrid memory, PCIe Gen5 NVMe SSD 등 추가 하드웨어 투자가 필요하며, staging buffer 크기와 DRAM cache 용량에 따라 비용-성능 trade-off가 존재한다[2, p.1][2, p.6][2, p.9][2, p.10][2, p.11].
- **D2(처리량·지연)**: Llama-3.1 8B/70B, 256 concurrent conversation 환경에서 recomputation-only baseline 대비 turn 5에서 최대 1.81× speedup(ideal GPU memory 80GB 대비 3.02×)을 보였다. remote CXL-hybrid memory의 latency로 인해 local GPU memory 대비 성능이 제한된다[2, p.9][2, p.10].
- **D3(품질 유지)**: 논문에서 정확도 벤치마크 결과나 출력 품질에 대한 수치적 평가가 보고되지 않아, 실제 품질 유지 여부는 판단 유보된다.
- **D4(도입 용이성)**: CXL-hybrid memory, PCIe Gen5 NVMe SSD, RDMA 네트워크 등 신규 하드웨어가 필요하고, staging buffer 등 운영 복잡도가 추가된다. 신규 인프라가 필요함이 명시된다[2, p.1][2, p.2][2, p.6][2, p.11].

#### 비교·대조

- SW는 비용 효율 및 처리량 측면에서 저자 보고 기준 수치적 개선이 확인되나, 동시성·지연·품질 손실에 대한 직접적 수치 근거는 부족하다. 도입 용이성은 구조적 변경이 필요하므로 중간 수준으로 평가된다.
- HW는 메모리 용량 한계 완화와 처리량 개선 잠재력이 있으나, 추가 하드웨어 도입과 운영 복잡도, 품질 유지에 대한 실측 근거 부족 등 한계가 존재한다.

---

## 5. 시사점

관점별로 평가가 엇갈리는 주요 지점은 다음과 같다.

| 주제 | 대상 | 시장 관점 | 도메인 관점 | 엇갈리는 이유 |
|---|---|---|---|---|
| 도입 용이성 및 추가 인프라 요구 | SW | 오픈소스 프레임워크 공식 지원, DeepSeek 계열 실서비스 배포로 도입 장벽 낮음 | 구조 변경 필요, 완전한 drop-in replacement 아님 | 시장성은 프레임워크 지원·배포 사례에 주목, 도메인은 실제 구조 변경·통합 난이도에 주목 |
| 비용 절감 효과의 실현 조건 | HW | CXL 메모리 확장 시장 성장, 일부 상용화 사례 | 추가 하드웨어 투자·운영 복잡도 필요 | 시장성은 시장 성장·상용화에 주목, 도메인은 실제 도입 시 인프라 투자·운영 복잡도에 주목 |
| 품질 및 정확도 영향 평가 | 공통 | 품질 영향에 대한 별도 우려 없음 | SW는 품질 비교 수치, HW는 정확도 벤치마크 부재로 한계 | 시장성은 품질 영향 근거 부재, 도메인은 실측 데이터 부재를 한계로 지적 |

기술 간 인식 차이는 SW는 소프트웨어 구조 변경만으로 효율성 개선을 추구하며, HW는 하드웨어 인프라 도입을 통해 자원 확장에 초점을 둔다는 점에서 뚜렷하다. SW는 기존 GPU/메모리 자원 내에서 효율을 극대화하고, HW는 자원 확장(메모리 용량 확대)에 집중한다.

두 방식은 병행 적용이 가능하다. SW(MLA)는 KV 캐시를 소프트웨어적으로 저차원 잠재 벡터로 압축하여 메모리 사용량을 줄이고, HW(ITME)는 CXL-hybrid memory 등 하드웨어 확장으로 대용량 메모리 공간을 제공한다. MLA 적용 모델의 KV 캐시를 ITME와 같은 계층형 메모리 구조에 오프로딩할 경우, 전체 시스템의 메모리 병목 완화 및 처리량 개선 효과가 상호 보완적으로 기대될 수 있다. 다만, 두 논문 모두 결합 효과를 직접 실험하지 않았으므로, 이는 검증되지 않은 가능성에 그친다[1, p.8][2, p.2].

SW와 HW 모두 저자 자체 실험 및 보고 수치에 의존하며, 독립적 대규모 상용 환경에서의 실측 데이터가 부족하다. MLA의 구조 변경이 대규모 분산 환경에서 미치는 영향, ITME의 실제 품질 유지 여부 등은 실증적 근거가 미흡하다. 시장성 평가의 수치는 기술 전체가 아닌 시장 전체 성장세에 대한 간접 근거임을 유의해야 한다. HW(ITME)는 추가 하드웨어 인프라 도입이 필수적이며, 실제 상용화·표준화 근거는 제한적이다.

---

## 6. 한계점

본 보고서는 공개 논문 및 웹 자료 기반으로 작성되었으며, 다음과 같은 한계가 존재한다.

- 저자 자체 실험 및 보고 수치에 의존: DeepSeek-V2 MLA, ITME 모두 저자 보고 기준의 실험 결과에 기반하므로, 독립적 대규모 상용 환경에서의 실측 데이터가 부족하다.
- 시장성 평가의 간접성: 시장 규모·성장성 등은 기술 전체가 아닌 시장 전체 성장세에 대한 간접 근거임을 명확히 하며, 기술별 시장 점유율·도입 기업 수 등 직접적 수치는 부족하다.
- 도메인 적용의 판단 유보: MLA 적용 전후 품질 손실, ITME의 정확도·출력 품질 등은 직접적 비교 수치가 없어 판단 유보 또는 한계로 기록했다.
- 도입 용이성의 불확실성: MLA의 구조 변경이 대규모 분산 환경에서 미치는 영향, ITME의 실제 운영 복잡도 등은 실증적 근거가 미흡하다.
- 표준화·생태계 확산 근거 부족: 두 기술 모두 공식 표준화 단체 채택, 복수 프레임워크 공식 지원 등 생태계 확산 근거가 제한적이다.
- 결합 효과의 미검증: MLA와 ITME의 병행 적용 효과는 두 논문 모두 직접 실험하지 않아, 검증되지 않은 가능성으로만 언급했다.
- 판단 유보 항목 명시: 근거가 부족한 경우 '판단 유보' 등급을 부여하여, 근거 없는 긍정·부정 평가를 피했다.

**확증편향 방지 조치**  
1. 질의 쌍 구성: 각 평가 기준마다 '채택·성과'와 '한계·제약(사실)' 질의 쌍으로 웹 검색을 수행, 의견·리뷰성 질의는 배제했다.  
2. 반대 근거 강제: 모든 기준 평가에 counter_evidence 필드를 필수로 두어, 지지 근거만 모으는 것을 구조적으로 차단했다.  
3. 근거 출처 구분: 논문 수치는 '저자 보고 기준'으로 표기하고, 제3자(웹) 자료와 구분, 인접 기술·일반 시장 자료는 '간접 근거'로 표시했다.  
4. 판단 유보 등급: 근거가 부족한 경우 '판단 유보' 등급을 부여했다.  
5. 모델 분리: 생성(Generator)과 판정(Judge)에 서로 다른 LLM을 사용해 자기 평가 편향을 줄였다.  
6. 이중 점검: 보고서 검토 단계에서 우열·추천 표현을 금칙어 규칙과 LLM Judge로 이중 검사했다.  
7. Reflection: 조사·평가 결과를 인용한 논문 페이지 원문과 다시 대조(LLM 검증 + 수치 정규식 대조)했다.

## REFERENCE

[1] DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv, 2405.04434.

[2] Jang, H. et al.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv, 2606.12556.

[3] pub.towardsai.net(n.d.). vLLM, SGLang, TRT-LLM: LLM Engine Guide. pub.towardsai.net, https://pub.towardsai.net/part-3-implementation-engine-level-choosing-the-runtime-that-gives-you-these-for-free-b0e9081205b0

[4] redhat.com(2025-03-05). Enhancing DeepSeek models with MLA and FP8 .... redhat.com, https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm

[5] OpenLM.ai(2024-12-26). DeepSeek-V3. OpenLM.ai, https://openlm.ai/deepseek-v3

[6] DeepWiki(2026-01-24). Multi-head Latent Attention (MLA) | deepseek-ai/DeepSeek-V3 | DeepWiki. DeepWiki, https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-(mla)

[7] Jesus Rodriguez(2025-08-27). The Sequence #710: Learning About DeepSeek v3.1 in 10 Key Points. thesequence.substack.com, https://thesequence.substack.com/p/the-sequence-710-learning-about-deepseek

[8] Dataintelo(2025-09-30). CXL Memory Module Market Research Report 2034. dataintelo.com, https://dataintelo.com/report/cxl-memory-module-market

[9] Spherical Insights(n.d.). CXL Memory Expansion Market Growth, Forecast Report 2035. Spherical Insights, https://www.sphericalinsights.com/reports/cxl-memory-expansion-market

[10] Nicolas(2026-05-20). Penguin Solutions ($PENG): Solving the AI Inference Memory Wall. snmart.substack.com, https://snmart.substack.com/p/penguin-solutions-peng-solving-the

[11] stocktitan.net(n.d.). Astera Labs Leo CXL Boost Azure M-series with 2TB | ALAB Stock News. stocktitan.net, https://www.stocktitan.net/news/ALAB/astera-labs-leo-cxl-smart-memory-controllers-on-microsoft-azure-m-dig12mr10z66.html

[12] Dataintelo(2025-06-28). Cross-Platform LLM Inference Engine Market. dataintelo.com, https://dataintelo.com/report/cross-platform-llm-inference-engine-market

[13] industryresearch.biz(n.d.). Large Language Model(LLM) Market - Size, Share & .... industryresearch.biz, https://www.industryresearch.biz/market-reports/large-language-model-llm-market-102241

[14] Global Market Insights Inc.(2025-09-01). Enterprise LLM Market Size & Share, Growth Analysis 2034. Global Market Insights Inc., https://www.gminsights.com/industry-analysis/enterprise-llm-market

[15] fortunebusinessinsights.com(n.d.). CXL Memory Pooling Appliance Market Size, Share [2026- .... fortunebusinessinsights.com, https://www.fortunebusinessinsights.com/cxl-memory-pooling-appliance-market-117960

[16] Amit Golander(2026-03-13). How CXL Memory Expansion Improves AI Economics. ASTERA LABS, INC., https://www.asteralabs.com/resources/blog/inference-tokenomics-how-cxl-memory-expansion-improves-ai-economics

[17] Introl(2026-03-29). CXL 4.0 Infrastructure Planning Guide | Introl Blog. Introl, https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025

[18] mlforsystems.org(n.d.). Exploring CXL-based KV Cache Storage for LLM Serving Yupeng Tang1∗. mlforsystems.org, https://mlforsystems.org/assets/papers/neurips2024/paper17.pdf

[19] ACL Anthology(n.d.). Enabling DeepSeek's Multi-Head Latent Attention in Any .... ACL Anthology, https://aclanthology.org/2025.acl-long.1597

[20] GitHub(n.d.). junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent .... GitHub, https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention

[21] Sebastian Raschka(n.d.). MLA Chapter 4 Guide | Sebastian Raschka, PhD. Sebastian Raschka, PhD, https://sebastianraschka.com/llms-from-scratch/ch04/05_mla
