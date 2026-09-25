# KV cache 최적화 기술 다관점 평가 보고서

**SW 압축 vs HW 메모리 확장** — DeepSeek-V2 MLA · ITME (CXL-Hybrid 계층 메모리 확장)  
평가 관점: 시장성 · 도메인 적용(데이터센터·클라우드 LLM 서빙) | 작성일: 2026-09-25 | 자동 생성(LangGraph Multi-Agent + Agentic RAG)

## SUMMARY

- SW(DeepSeek-V2 MLA)와 HW(ITME)는 모두 대규모 LLM 추론 서빙 시장의 성장성과 수요 증가에 기반해 시장성 측면에서 긍정적으로 평가된다. 그러나 두 기술 모두 실제 상용화·생태계 통합 수준은 초기 단계에 머무르며, 외부·독립적 검증 데이터가 부족하다.
- 도메인 적용 관점에서 SW(MLA)는 소프트웨어적 구조 변경만으로 KV 캐시를 압축해 처리량 증가와 비용 효율화 가능성을 보이나, 극단적 압축 시 품질 저하 가능성 및 실제 도입 난이성에 대한 근거가 부족하다. HW(ITME)는 하드웨어 확장으로 대규모 메모리 공간을 제공하지만, 추가 인프라 요구와 품질 유지에 대한 벤치마크 부재가 한계로 지적된다.
- 가장 큰 상충 지점은 ‘도입 난이도 및 인프라 요구사항’과 ‘정확도·출력 품질 유지’에서 나타난다. 시장성 평가는 성장성·확산 가능성에 초점을 두는 반면, 도메인 평가는 실제 인프라 요구와 품질 유지에 더 엄격한 기준을 적용한다.
- 두 기술은 KV 캐시 병목 해소라는 공통 목표를 지향하며, SW(MLA)로 캐시 크기를 줄이고 HW(ITME)로 메모리 공간을 확장하는 상호 보완적 가능성이 있으나, 결합 효과는 아직 검증되지 않았다.

---

## 1. 분석 배경

대규모 언어모델(LLM) 추론에서 연산 병목이 점차 메모리(HBM 용량) 병목으로 전환되고 있다. 이는 LLM의 컨텍스트 길이와 동시 요청 수가 증가함에 따라, KV(Key-Value) 캐시가 GPU 메모리의 주요 자원 소모원이 되었기 때문이다. 이에 따라, KV 캐시 병목을 해소하기 위한 기술은 크게 두 진영으로 나뉜다. 첫째, 소프트웨어(SW) 진영은 어텐션 아키텍처 개선 등으로 데이터를 작게 만들어 메모리 사용량을 줄이는 방식을 추구한다. 둘째, 하드웨어(HW) 진영은 CXL 기반 메모리 확장 등으로 공간을 넓혀 근본적으로 용량 한계를 완화한다. 이 두 접근법은 상호 대체적이기보다는 병행될 수 있으며, 실제로 데이터센터·클라우드 LLM 서빙 환경에서는 두 방식의 조합이 논의되고 있다.

본 보고서는 데이터센터·클라우드 LLM 서빙을 평가 도메인으로 삼고, 시장성 관점(시장 규모·성장성, 상용화·채택 현황, 생태계 지지)과 도메인 적용 관점(비용 효율, 처리량·지연, 품질 유지, 도입 용이성)에서 SW·HW 대표 기술을 다관점으로 비교·대조한다. 기술성숙도, 이해관계자(경쟁사·개발자·투자자) 관점은 평가 범위에서 제외한다.

---

## 2. 기술 선정

본 평가는 에이전트 기반 선정 대신 Human 기반 방식으로, 과제 Doc Pool 에서 SW·HW 진영별 1개 기술을 직접 선정했다.

- **SW(어텐션 아키텍처 개선): DeepSeek-V2 MLA**  
  선정 사유: KV cache를 사후 압축하지 않고 어텐션 구조(Multi-head Latent Attention) 자체를 바꿔 KV cache 93.3% 감소를 보고한 기술이다. 공개 모델·오픈소스 서빙 스택 채택 사례가 있어 시장성 관점의 근거가 풍부하고, 재학습이 필요한 구조 변경이라 도메인 도입 관점과 대비가 뚜렷하다.

- **HW(메모리/스토리지 계층 확장): ITME (CXL-Hybrid 계층 메모리 확장)**  
  선정 사유: 원문 KV를 그대로 두고 CXL-Hybrid 메모리로 TB급 바이트 주소 공간을 확장하는 방식으로, 배경에서 제시된 'CXL 메모리 방식' 진영을 대표한다. 데이터센터 공유 컨텍스트 인프라를 직접 겨냥하며, 2026년 발표된 메모리 기업(SK hynix) 연구라 시장 근거가 간접적이라는 점 자체가 관점 간 차이를 드러낸다.

---

## 3. 기술 개요

### DeepSeek-V2 MLA (SW · 어텐션 아키텍처 개선)

- **핵심 방식**:  
  DeepSeek-V2 MLA는 기존 Transformer의 Multi-Head Attention(MHA) 대비 추론 효율성을 높이기 위해, Key-Value(KV) 캐시를 잠재 벡터로 압축하는 Multi-head Latent Attention(MLA) 구조를 도입한다. low-rank key-value joint compression, RoPE positional encoding, 쿼리·키 분리, KV 캐시 차원 축소 등 수식 기반으로 구현된다. [1, p.6][1, p.8]

- **저자 보고 성과**:  
  - DeepSeek 67B 대비 훈련 비용 42.5% 절감, KV 캐시 93.3% 감소, 최대 생성 처리량 5.76배 증가(8 H800 GPU 노드 기준) [1, p.1][1, p.16][1, p.21]
  - 21B 활성화 파라미터만으로 오픈소스 모델 중 최고 수준의 성능 달성(정확도 벤치마크) [1, p.21]

- **한계**:  
  - 사전학습 이후 지식 업데이트 없음, 비사실적 정보 생성 가능성 [1, p.21]
  - 주로 중국어·영어 데이터로 학습되어 타 언어 성능 제한 [1, p.21]
  - MLA 적용을 위해서는 KV 캐시를 잠재 벡터로 압축하는 연산 및 DeepSeek-V2/MLA 아키텍처 지원 필요 [1, p.6][1, p.29]

---

### ITME (HW · CXL-Hybrid 계층 메모리 확장)

- **핵심 방식**:  
  ITME는 CXL-hybrid memory 및 PCIe Gen5 NVMe SSD를 활용해 TB급의 byte-addressable 원격 메모리 확장을 제공한다. 내부 하드웨어 프리페처와 사용자 수준 prefetcher API를 통해 LLM 추론 프레임워크가 데이터 이동을 명시적으로 제어하며, pipelined multi-tier DMA 기반 prefetching으로 전체 지연을 가린다. [2, p.2][2, p.3][2, p.7][2, p.8]

- **저자 보고 성과**:  
  - production-grade SK Hynix CMM 및 PCIe Gen5 NVMe SSD 환경에서 최대 35.7% throughput 개선 [2, p.1][2, p.11]
  - Llama-3.1 8B/70B, 256 concurrent conversation, 35-turn benchmark에서 recomputation 기반 baseline 대비 최대 1.81× speedup(5번째 turn) [2, p.9]
  - 모델 weight prefetching만 적용한 실험에서 host memory baseline과 유사한 end-to-end 성능 [2, p.10]

- **한계**:  
  - remote CXL-hybrid memory의 latency에 의해 local GPU memory 대비 성능 제한 [2, p.9]
  - staging buffer 크기에 따라 host-side resource overhead 증가 가능 [2, p.6]
  - LLM workload의 결정적 접근 패턴에 의존 [2, p.2][2, p.3]
  - CXL-hybrid memory, PCIe Gen5 NVMe SSD, RDMA 네트워크 등 추가 하드웨어 필요 [2, p.2][2, p.11]

---

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| M1 시장 규모·성장성 | 높음 (신뢰도 중간) | 높음 (신뢰도 중간) |
| M2 상용화·채택 현황 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |
| M3 생태계 지지 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |

#### M1 시장 규모·성장성

- **SW(MLA)**:  
  LLM 추론 서빙 시장은 2025~2035년 연평균 21~25%의 고성장이 예측되며, MLA와 같은 추론 효율화 기술이 속한 시장의 성장성이 뒷받침된다. 다만, MLA 자체의 시장 점유율·직접적 수요 수치는 없다(간접 근거).  
  - 근거: [3][4][5]
  - 반대 근거: MLA 기술 자체의 시장 점유율, 실제 도입 규모 등 직접적 수요 데이터는 없음

- **HW(ITME)**:  
  CXL-hybrid memory 및 메모리 확장 시장은 2025년 28억 달러에서 2034년 286억 달러로 연평균 29% 이상 성장 전망. AI, LLM 추론, 데이터센터가 주요 수요처로 반복 언급된다. ITME 자체 시장 수치는 없으므로 간접 근거임을 명시.  
  - 근거: [6][7][8]
  - 반대 근거: CXL 확장 비용, 기존 시스템과의 호환성, 인프라 업그레이드 비용 등 도입 장벽 존재

#### M2 상용화·채택 현황

- **SW(MLA)**:  
  DeepSeek-V2 MLA는 실제 서비스에 배포되어 운영 중임이 보고되고, 오픈소스 모델(DeepSeek-V2-Lite)로도 공개되어 있다. vLLM, SGLang 등 주요 LLM 서빙 프레임워크에서 지원된다. 그러나 AWS, Azure, GCP 등 주요 상용 클라우드 서비스에서 공식 지원·운영된다는 독립적 3자 근거는 부족하다.  
  - 근거: [1, p.16][1, p.5][9][10][11]
  - 반대 근거: 대규모 엔터프라이즈 도입 사례, 상용화 실적에 대한 외부 리포트 부재

- **HW(ITME)**:  
  ITME는 FPGA 기반 하드웨어 프로토타입과 production-grade 환경에서 실험적으로 구현·평가되었다. 상용 제품/서비스로의 공식 출시, 대규모 운영 사례, 오픈소스 메인라인 통합 등은 확인되지 않는다.  
  - 근거: [2, p.2][2, p.11][12][13][14]
  - 반대 근거: ITME의 상용 제품 출시, 대규모 서비스 운영, 오픈소스 메인라인 통합 등은 확인되지 않음

#### M3 생태계 지지

- **SW(MLA)**:  
  DeepSeek-V2, V3, Kimi K2 등 복수 오픈소스 LLM에서 MLA 채택, vLLM, SGLang 등 주요 LLM 서빙 프레임워크에서 지원. 업계 표준(ONNX, HuggingFace 등)이나 표준화 단체에서 공식 채택된 사례는 없음.  
  - 근거: [9][10][11][15]
  - 반대 근거: 표준화 단체(MLCommons, IEEE 등)에서 MLA를 공식 표준으로 채택했다는 근거 없음

- **HW(ITME)**:  
  CXL-hybrid memory 및 메모리 확장 기술은 vLLM, SGLang, TensorRT-LLM 등에서 KV cache 오프로딩, paged KV cache 등으로 실험적/부분적 지원. ITME 자체가 공식 표준에 채택되거나 복수 프레임워크에서 공식 지원된 근거는 없음.  
  - 근거: [16][17][18][19][20][6][8]
  - 반대 근거: ITME 자체가 복수 프레임워크에서 공식 지원되거나, 표준화 단체에 채택된 근거는 없음

---

### 4.2 도메인 적용 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| D1 비용 효율 | 중간 (신뢰도 높음) | 중간 (신뢰도 중간) |
| D2 처리량·지연 | 중간 (신뢰도 높음) | 중간 (신뢰도 중간) |
| D3 품질 유지 | 중간 (신뢰도 중간) | 판단 유보 (신뢰도 낮음) |
| D4 도입 용이성 | 판단 유보 (신뢰도 중간) | 낮음 (신뢰도 중간) |

#### D1 비용 효율

- **SW(MLA)**:  
  DeepSeek-V2는 DeepSeek 67B 대비 훈련 비용 42.5% 절감, KV 캐시 93.3% 감소, 최대 생성 처리량 5.76배 증가(저자 보고 기준). 동일 워크로드에서 GPU·메모리 비용 절감이나 추가 비용 없이 수용량 확대가 실제로 수치로 확인된 근거는 없음.  
  - 근거: [1, p.1][1, p.16][1, p.21]
  - 반대 근거: 실제 비용 절감 수치의 외부 검증 부재

- **HW(ITME)**:  
  ITME는 production-grade 환경에서 최대 35.7% throughput 개선(저자 보고 기준). staging buffer 크기에 따라 host-side resource overhead가 증가할 수 있음. 실제 GPU·HBM 비용 절감 수치는 직접적으로 제시되지 않음.  
  - 근거: [2, p.1][2, p.6][2, p.11]
  - 반대 근거: 실제 비용 절감 수치 부재

#### D2 처리량·지연

- **SW(MLA)**:  
  8 H800 GPU 단일 노드 기준, 최대 생성 처리량 5.76배 증가(저자 보고 기준). latency(지연)나 SLO 유지에 대한 수치 근거는 없음.  
  - 근거: [1, p.16][1, p.21]
  - 반대 근거: latency·SLO 유지에 대한 수치 근거 부재

- **HW(ITME)**:  
  recomputation 기반 baseline 대비 최대 1.81× speedup(5번째 turn), 최대 35.7% throughput 개선(저자 보고 기준). local GPU memory 대비 latency 측면에서 한계, remote CXL-hybrid memory의 latency에 따라 성능 제한.  
  - 근거: [2, p.9][2, p.10][2, p.11]
  - 반대 근거: 모든 조건에서 latency가 유지된다는 근거 부족

#### D3 품질 유지

- **SW(MLA)**:  
  DeepSeek-V2는 21B 활성화 파라미터만으로 오픈소스 모델 중 최고 수준의 성능 달성(저자 보고 기준). MLA 구조상 품질 손실이 없다는 설계적 단정이나, 모든 벤치마크에서 손실 없음이 수치로 명확히 제시된 근거는 없음.  
  - 근거: [1, p.15][1, p.21]
  - 반대 근거: 극단적 압축 시 품질 저하 가능성에 대한 추가 벤치마크 미확인

- **HW(ITME)**:  
  논문 내 정확도·출력 품질에 대한 벤치마크 결과가 직접적으로 제시되지 않아 판단 유보.  
  - 근거: 없음
  - 반대 근거: 정확도·출력 품질 벤치마크 부재

#### D4 도입 용이성

- **SW(MLA)**:  
  MLA는 기존 MHA 대비 아키텍처 변경이 필요하며, KV 캐시를 잠재 벡터로 압축하는 연산이 추가된다. 실제로 기존 서빙 스택에 일부 커널·시스템 수정이 필요한지, 완전한 drop-in replacement가 아닌지에 대한 직접적 근거는 없음.  
  - 근거: [1, p.6][1, p.29]
  - 반대 근거: 도입 난이성에 대한 직접적 근거 부족

- **HW(ITME)**:  
  CXL-hybrid memory, PCIe Gen5 NVMe SSD, RDMA 네트워크 등 추가 하드웨어 필요. 일부 커널·시스템 수정 및 제한적 HW 추가가 요구되므로, 완전한 SW-only 도입은 아니다.  
  - 근거: [2, p.2][2, p.5][2, p.11]
  - 반대 근거: 실제 대규모 상용 배포 사례 부재

---

## 5. 시사점

관점별 평가에서 SW(MLA)와 HW(ITME)는 시장성 측면에서는 모두 성장하는 LLM 추론 서빙 시장의 흐름과 맞닿아 있으나, 도메인 적용에서는 도입 난이도, 품질 유지, 실제 상용화 수준 등에서 평가가 엇갈린다.

| 주제 | 대상 | 시장 관점 | 도메인 관점 | 엇갈리는 이유 |
|---|---|---|---|---|
| 도입 난이도 및 인프라 요구사항 | 공통 | 성장하는 시장 내 도입 가능성에 초점, 도입 장벽 언급 제한적 | SW: 일부 시스템 수정 필요성 근거 부족, HW: 추가 하드웨어·시스템 수정 필요 명확 | 시장성은 성장성·확산 가능성 중시, 도메인은 실제 인프라 요구·도입 난이성 중시 |
| 정확도·출력 품질 유지 | 공통 | 성능·효율화에 초점, 품질 저하 언급 제한적 | SW: 일부 벤치마크에서 품질 손실 없음 미확인, HW: 품질 벤치마크 부재로 판단 유보 | 시장성은 효율성·성장성 중시, 도메인은 품질 유지·신뢰성 중시 |
| 상용화 및 생태계 통합 수준 | 공통 | 오픈소스 프레임워크 지원, PoC/실증 사례로 긍정 평가 | 실제 상용 클라우드 서비스 공식 지원, 대규모 엔터프라이즈 도입 등 구체적 증거 부족 | 시장성은 실험적 지원도 확산 신호로 간주, 도메인은 실제 운영·통합 증거 요구 |

기술 간 인식 차이는 SW(MLA)가 소프트웨어적 구조 변경만으로 효율화를 추구하는 반면, HW(ITME)는 하드웨어 인프라 확장이 필수라는 점에서 도입 난이성, 인프라 요구, 품질 유지 등에서 뚜렷하게 드러난다. 두 기술은 KV 캐시 병목 해소라는 공통 목표를 지향하며, SW(MLA)로 캐시 크기를 줄이고 HW(ITME)로 확장된 메모리 공간을 활용하면 대규모 LLM 서빙에서 처리량·동시성 극대화에 상호 보완적으로 기여할 수 있다. 다만, 두 논문 모두 결합 효과를 직접 실험하지 않았으므로, 결합 효과는 검증되지 않은 가능성에 머문다. [1, p.6][2, p.2]

---

## 6. 한계점

본 보고서는 공개 논문 및 웹 자료 기반으로 작성되었으며, 다음과 같은 한계가 있다.

- 저자 자체 실험 및 보고에 의존하는 수치가 많아, 외부·독립적 검증 데이터가 부족하다.
- MLA 및 ITME 모두 실제 상용 클라우드 서비스, 다양한 하드웨어 환경, 대규모 엔터프라이즈 운영 사례에 대한 정보가 제한적이다.
- ITME는 정확도·출력 품질에 대한 벤치마크가 논문에 제시되지 않아, 품질 유지 여부에 대한 판단이 어렵다.
- MLA는 극단적 KV 캐시 압축 시 품질 저하 가능성에 대한 추가 벤치마크가 필요하다.
- 두 기술 모두 표준화·생태계 통합 수준이 초기 단계로, 향후 업계 표준 채택 여부에 따라 실제 확산 속도가 달라질 수 있다.
- 시장성 평가는 시장 규모·성장성 등 간접 근거에 의존하며, 실제 도입 규모·점유율 등 직접적 수치가 부족하다.
- 도메인 적용 평가는 실제 비용 절감, 품질 유지, 도입 난이성 등에서 판단 유보 항목이 존재한다.

**확증편향 방지 조치**  
1. 질의 쌍 구성: 기준마다 웹 검색을 '채택·성과'와 '한계·제약(사실)' 질의 쌍으로 수행, 의견·리뷰 질의는 배제  
2. 반대 근거 강제: 모든 기준 평가에 counter_evidence 필드를 필수로 두어 지지 근거만 모으는 것을 구조적으로 차단  
3. 근거 출처 구분: 논문 수치는 '저자 보고 기준'으로 표기, 제3자(웹) 자료와 구분, 인접 기술·일반 시장 자료는 '간접 근거'로 표시  
4. 판단 유보 등급: 근거가 부족하면 등급을 만들지 않고 '판단 유보'로 기록  
5. 모델 분리: 생성(Generator)과 판정(Judge)에 서로 다른 LLM을 사용해 자기 평가 편향을 줄임  
6. 이중 점검: 보고서 검토 단계에서 우열·추천 표현을 금칙어 규칙과 LLM Judge로 이중 검사  
7. Reflection: 조사·평가 결과를 인용한 논문 페이지 원문과 다시 대조(LLM 검증 + 수치 정규식 대조)

## REFERENCE

### 논문

[1] DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434.

[2] Jang, H. et al.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556.

### 기타

[3] Dataintelo(2025-06-28). *Cross-Platform LLM Inference Engine Market Research Report 2034*. dataintelo.com, https://dataintelo.com/report/cross-platform-llm-inference-engine-market

[4] industryresearch.biz(n.d.). *Large Language Model(LLM) Market - Size, Share & Industry Forecast*. industryresearch.biz, https://www.industryresearch.biz/market-reports/large-language-model-llm-market-102241

[5] Global Market Insights Inc.(2025-09-01). *Enterprise LLM Market Size & Share, Growth Analysis 2034*. Global Market Insights Inc., https://www.gminsights.com/industry-analysis/enterprise-llm-market

[6] Dataintelo(2025-09-30). *CXL Memory Module Market Research Report 2034*. dataintelo.com, https://dataintelo.com/report/cxl-memory-module-market

[7] Spherical Insights(n.d.). *CXL Memory Expansion Market Growth, Forecast Report 2035*. Spherical Insights, https://www.sphericalinsights.com/reports/cxl-memory-expansion-market

[8] snia.org(n.d.). *The Rebirth of Persistent Memory: How CXL lays*. snia.org, https://www.snia.org/blog/2026/rebirth-persistent-memory-how-cxl-lays-foundation-current-and-next-generation-persistent

[9] hungchun0201.github.io(n.d.). *DeepSeek-V2: Multi-Head Latent Attention (MLA)*. hungchun0201.github.io, https://hungchun0201.github.io/agentic-ai-survey/papers/deepseek-mla/index.html

[10] Spheron(2026-06-25). *Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache ~98% and Serve More Users Per GPU (2026 Guide) | Spheron Blog*. Spheron, https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud

[11] emergentmind.com(n.d.). *DeepSeek MLA: Scalable Latent Attention*. emergentmind.com, https://www.emergentmind.com/topics/deepseek-mla

[12] Samsung Semiconductor Global(2026-09-04). *Breaking AI Memory Limits with CXL Memory Pooling*. Samsung Semiconductor Global, https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling

[13] Sandeep Dattaprasad(2025-11-20). *How CXL Transforms RAG and KV Cache Performance*. ASTERA LABS, INC., https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance

[14] Sandeep Dattaprasad(2025-11-20). *How CXL Transforms RAG and KV Cache Performance*. ASTERA LABS, INC., https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance

[15] Sebastian Raschka(n.d.). *Multi-Head Latent Attention (MLA)*. Sebastian Raschka, PhD, https://sebastianraschka.com/llms-from-scratch/ch04/05_mla

[16] Amit Golander(2026-03-13). *How CXL Memory Expansion Improves AI Economics*. ASTERA LABS, INC., https://www.asteralabs.com/resources/blog/inference-tokenomics-how-cxl-memory-expansion-improves-ai-economics

[17] Philip Kiely(2026). *vLLM vs SGLang vs TensorRT-LLM - Inference Engineering*. inferenceengineering.tech, https://inferenceengineering.tech/learn/vllm-vs-sglang-vs-tensorrt-llm

[18] runpod.io(2026-06-23). *vLLM vs TensorRT-LLM vs SGLang: Which Inference Engine to Deploy*. runpod.io, https://www.runpod.io/articles/comparison/vllm-vs-tensorrt-llm

[19] medium.com(n.d.). *How We Solved the KV Cache Bottleneck in LLM Inference*. medium.com, https://medium.com/@jooho.lee_93960/how-we-solved-the-kv-cache-bottleneck-in-llm-inference-with-cxl-shared-memory-d5133dcfb044

[20] mlforsystems.org(n.d.). *Exploring CXL-based KV Cache Storage for LLM Serving Yupeng Tang1∗*. mlforsystems.org, https://mlforsystems.org/assets/papers/neurips2024/paper17.pdf
