# KV cache 최적화 기술 다관점 평가 보고서

**SW 압축 vs HW 메모리 확장** — DeepSeek-V2 MLA · ITME (CXL-Hybrid 계층 메모리 확장)  
평가 관점: 시장성 · 도메인 적용(데이터센터·클라우드 LLM 서빙) | 작성일: 2026-09-25 | 자동 생성(LangGraph Multi-Agent + Agentic RAG)

## SUMMARY

- 시장성 관점에서 HW(CXL-hybrid 계층 메모리 확장, ITME)는 전체 CXL 메모리 확장 시장의 높은 성장성(2025~2035년 연평균 29% 이상)과 AI/LLM 추론에서의 TB급 메모리 수요 급증에 힘입어 긍정적으로 인식된다. SW(DeepSeek-V2 MLA)는 LLM 추론 서빙 시장의 성장과 오픈소스·프레임워크 채택 확산이 확인되나, 직접적 시장 규모 및 대규모 상용 서비스 도입 근거는 제한적이다.
- 도메인 적용 관점에서는 SW(MLA)가 KV 캐시 93.3% 감소, 최대 5.76배 처리량 증가 등 비용 효율성·처리량 개선 효과를 저자 기준으로 보고하나, 품질 손실 및 latency(SLO) 유지에 대한 독립적 검증이 부족하다. HW(ITME)는 TB급 메모리 확장과 throughput 최대 35.7% 개선을 주장하지만, 실제 비용 절감 수치·정확도 벤치마크가 미흡하고, 하드웨어 추가·운영 복잡성 등 도입 장벽이 강조된다.
- 가장 큰 상충 지점은 HW 기술의 시장성(성장성·수요)과 도메인 적용(실질 도입 장벽) 간 인식 차이, SW 기술의 프레임워크 확장성(시장성)과 실제 대규모 운영 환경에서의 호환성·운영 리스크(도메인 적용) 간 온도차다.
- 두 기술 모두 저자 자체 실험 및 간접 시장 자료에 의존하며, 제3자 독립 벤치마크 및 대규모 상용 서비스 도입 근거가 부족하다는 점에서 한계가 공통적으로 지적된다.

---

## 1. 분석 배경

KV cache는 대규모 언어 모델(LLM) 추론에서 연산 병목을 메모리(HBM 용량) 병목으로 전환시키는 핵심 요인으로 부상했다. 기존에는 연산량이 주요 제약이었으나, 긴 컨텍스트와 대규모 동시 요청이 일반화되면서 KV cache가 GPU HBM의 공간을 빠르게 소진해 처리량·비용 효율성 저하를 초래한다. 이에 따라, SW(데이터를 작게)와 HW(공간을 넓게) 두 진영이 각각 어텐션 구조 혁신과 메모리 계층 확장 방식으로 대응하고 있다. 이 두 접근법은 상호 대체가 아니라 병행 적용이 가능하며, 실제 일부 LLM 서빙 프레임워크에서는 SW/HW 병행 최적화 연구가 진행 중이다.

본 보고서는 데이터센터·클라우드 LLM 서빙을 평가 도메인으로 삼고, 시장성(시장 규모·채택·생태계)과 도메인 적용(비용 효율·성능·품질·도입 용이성) 두 관점에서 기술을 비교·대조한다. 기술성숙도, 이해관계자(경쟁사·개발자·투자자) 관점은 평가 범위에 포함하지 않는다.

---

## 2. 기술 선정

본 평가는 에이전트 기반 선정 대신 Human 기반 방식으로, 과제 Doc Pool 에서 SW·HW 진영별 1개 기술을 직접 선정했다.

- **SW: DeepSeek-V2 MLA**  
  선정 사유: KV cache를 사후 압축하지 않고 어텐션 구조(Multi-head Latent Attention) 자체를 바꿔 KV cache 93.3% 감소를 보고한 기술이다. 공개 모델·오픈소스 서빙 스택 채택 사례가 있어 시장성 관점의 근거가 풍부하고, 재학습이 필요한 구조 변경이라 도메인 도입 관점과 대비가 뚜렷하다.

- **HW: ITME (CXL-Hybrid 계층 메모리 확장)**  
  선정 사유: 원문 KV를 그대로 두고 CXL-Hybrid 메모리로 TB급 바이트 주소 공간을 확장하는 방식으로, 'CXL 메모리 방식' 진영을 대표한다. 데이터센터 공유 컨텍스트 인프라를 직접 겨냥하며, 2026년 발표된 메모리 기업(SK hynix) 연구라 시장 근거가 간접적이라는 점 자체가 관점 간 차이를 드러낸다.

---

## 3. 기술 개요

### DeepSeek-V2 MLA (SW · 어텐션 아키텍처 개선)

- **핵심 방식**:  
  Multi-head Latent Attention(MLA)은 기존 Transformer의 Multi-Head Attention(MHA) 대비, low-rank key-value joint compression을 활용해 KV 캐시를 잠재 공간(latent space)으로 압축한다. 이를 통해 캐시 오버헤드를 대폭 줄이고, 추론 효율성을 높인다. MLA의 구조와 수식은 논문 본문 및 부록에 상세히 제시되어 있다. [1, p.6][1, p.8]

- **저자 보고 성과**:  
  DeepSeek-V2는 236B 파라미터(토큰당 21B 활성화), 128K 컨텍스트 길이를 지원하며, MLA와 DeepSeekMoE 결합으로 DeepSeek 67B 대비 학습 비용 42.5% 절감, KV 캐시 93.3% 감소, 최대 생성 처리량 5.76배 증가를 달성했다고 보고했다. MLA는 GQA(2.25 groups)와 유사한 수준의 KV 캐시만 필요하지만, MHA보다 더 강력한 성능을 보인다고 주장한다. [1, p.1][1, p.6][1, p.8][1, p.21]

- **한계**:  
  DeepSeek-V2 및 Chat 버전은 사전학습 이후 지속적 지식 업데이트가 없으며, 비사실적 정보 생성 가능성이 있다. 학습 데이터가 주로 중국어·영어로 구성되어 기타 언어 성능이 제한될 수 있다. MLA 적용을 위해서는 low-rank key-value joint compression을 지원하는 아키텍처와 MLA 수식 구현이 필요하다. [1, p.21][1, p.6][1, p.8]

---

### ITME (HW · CXL-Hybrid 계층 메모리 확장)

- **핵심 방식**:  
  ITME는 CXL-hybrid memory 기반 계층형 메모리 확장 구조와, 결정적 접근 패턴을 활용한 multi-tier DMA 기반 prefetching 파이프라인을 제공한다. 내부 하드웨어 prefetcher가 SSD에서 DRAM cache로 데이터를 이동시키고, 사용자 수준 prefetcher API를 통해 LLM 추론 프레임워크가 데이터 이동을 명시적으로 제어한다. RDMA와 호스트 CPU 메모리를 통한 파이프라인화된 데이터 이동으로 GPU 연산과 데이터 전송을 중첩시켜 전체 지연을 가린다. [2, p.2][2, p.3][2, p.7][2, p.8]

- **저자 보고 성과**:  
  ITME는 대규모 KV cache footprint를 호스트 메모리 한계를 넘어 확장할 수 있으며, 최대 35.7%의 throughput 개선을 달성했다고 보고했다. Llama-3.1 8B/70B 모델 기준 recomputation 대비 turn 5에서 최대 1.81×의 normalized TTFT speedup을 보였다. multi-turn inference에서 CPU-offload baseline과 초반(turn 1–9)에는 유사한 성능, 이후(turn 10–20)에는 I/O contention으로 성능 하락이 관찰됐다. [2, p.1][2, p.9][2, p.10][2, p.11]

- **한계**:  
  remote CXL-hybrid memory의 latency 한계로 local GPU memory 대비 성능이 제한된다. host memory staging buffer 크기에 따라 I/O resilience와 메모리 오버헤드 간 trade-off가 존재한다. ITME의 성능은 LLM workload의 결정적 접근 패턴에 의존한다. FPGA 기반 프로토타입으로 기능적 실현 가능성은 입증됐으나, 대규모 상용 운영 사례는 부족하다. [2, p.1][2, p.2][2, p.3][2, p.6][2, p.9][2, p.11]

---

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| M1 시장 규모·성장성 | 중간 (신뢰도 중간) | 높음 (신뢰도 중간) |
| M2 상용화·채택 현황 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |
| M3 생태계 지지 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |

#### SW: DeepSeek-V2 MLA

- **시장 규모·성장성(M1)**:  
  MLA가 겨냥하는 LLM 추론 서빙 및 인접 시장은 2025~2035년 연평균 21~25% 성장 전망이 제시된다. 다만, 'KV 캐시 최적화' 세부 시장 규모는 별도 집계되어 있지 않아 간접 근거임을 감안해 '중간'으로 평가된다.  
  *근거*: 글로벌 cross-platform LLM inference engine 시장 2025년 38억 달러→2034년 286억 달러(25.2% CAGR) [4], 엔터프라이즈 LLM 시장 2025년 88억 달러→2034년 711억 달러(25.8% CAGR) [5], LLM 전체 시장 2026~2035년 21.62% CAGR [6].  
  *반대 근거*: 시장조사기관별 범위 상이, MLA와 직접 연결된 세부 시장 규모 미집계.

- **상용화·채택 현황(M2)**:  
  MLA는 DeepSeek-V2, V2-Lite 등 모델에 적용되어 논문·오픈소스로 공개되어 있고, 일부 추론 엔진(SGLang, LMDeploy, TensorRT-LLM 등)에서 지원된다. 그러나 글로벌 대규모 상용 서비스 도입 근거는 부족하다.  
  *근거*: DeepSeek-V2, V2-Lite 등 MLA 적용 모델 오픈소스 공개 [1, p.1][1, p.5][7], 일부 프레임워크 지원 [8][9][10].  
  *반대 근거*: 주요 클라우드 사업자 공식 상용 서비스 채택 근거 부족.

- **생태계 지지(M3)**:  
  MLA는 복수 프레임워크에서 지원 언급이 있으나, 공식 산업 표준(ONNX, MLPerf 등) 채택 사례는 없다.  
  *근거*: SGLang, LMDeploy, TensorRT-LLM, vLLM 등에서 지원 언급 [8][9][10], 다양한 모델 적용 [11][12].  
  *반대 근거*: 공식 표준 채택·규격화 사례 미확인.

#### HW: ITME (CXL-Hybrid 계층 메모리 확장)

- **시장 규모·성장성(M1)**:  
  CXL 기반 메모리 확장 시장은 2025년 10억~28억 달러, 2034~2035년까지 연평균 29% 이상 성장 전망이 제시된다. AI/LLM 추론에서 TB급 메모리 수요 급증이 시장 성장성을 뒷받침한다.  
  *근거*: CXL 메모리 모듈 시장 2025년 28억 달러→2034년 286억 달러(29.4% CAGR) [13], CXL 메모리 확장 시장 2025년 12.7억 달러→2035년 164.6억 달러(29.2% CAGR) [14].  
  *반대 근거*: 기관별 수치 편차, 실제 대규모 상용화는 초기 단계.

- **상용화·채택 현황(M2)**:  
  ITME는 논문 및 FPGA 프로토타입 수준에서 기능적 실현 가능성이 입증됐으나, 상용 제품·대규모 서비스 도입 사례는 없다. 인접 기술(CXL 기반 KV cache 오프로딩 등)은 주요 LLM 서빙 스택에 통합되고, 여러 업체에서 데모·평가가 이루어지고 있다.  
  *근거*: ITME FPGA 프로토타입 [2, p.11], vLLM, LMCache 등 주요 스택 통합 [15], XConn, MemVerge, Samsung 등 데모 [16][17][18].  
  *반대 근거*: ITME 고유 상용 제품·운영 사례 미확인.

- **생태계 지지(M3)**:  
  ITME는 vLLM 기반 연동이 가능하나, 고유 표준 채택·복수 프레임워크 공식 지원 사례는 없다.  
  *근거*: user-level prefetcher API, vLLM 연동 [2, p.2][2, p.1], CXL 기반 KV cache 오프로딩 주요 프레임워크 통합 [15][16].  
  *반대 근거*: ITME 고유의 공식 표준·복수 프레임워크 지원 미확인.

---

### 4.2 도메인 적용 관점

| 기준 | SW: DeepSeek-V2 MLA | HW: ITME (CXL-Hybrid 계층 메모리 확장) |
|---|---|---|
| D1 비용 효율 | 높음 (신뢰도 높음) | 중간 (신뢰도 중간) |
| D2 처리량·지연 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |
| D3 품질 유지 | 중간 (신뢰도 중간) | 판단 유보 (신뢰도 낮음) |
| D4 도입 용이성 | 중간 (신뢰도 중간) | 중간 (신뢰도 중간) |

#### SW: DeepSeek-V2 MLA

- **비용 효율(D1)**:  
  KV 캐시 93.3% 감소, 최대 생성 처리량 5.76배 증가, 학습 비용 42.5% 절감 등 비용 효율성 개선이 저자 기준으로 보고된다. 실제 서빙 환경에서 더 큰 배치 사이즈를 처리할 수 있다.  
  *근거*: [1, p.1][1, p.16][1, p.21]  
  *반대 근거*: 제3자 독립 검증 자료 없음, 추가 SW 구현·운영 복잡성 가능성.

- **처리량·지연(D2)**:  
  8 H800 GPU 단일 노드에서 초당 50K 토큰 이상 생성(DeepSeek 67B 대비 5.76배 증가)이 보고되나, latency(지연) 수치나 SLO 충족 여부에 대한 직접적·정량적 근거는 없다.  
  *근거*: [1, p.16][1, p.1][1, p.21]  
  *반대 근거*: 지연 절대값·SLO 충족 여부 미확인, 일반화 추가 검증 필요.

- **품질 유지(D3)**:  
  주요 벤치마크에서 동급 또는 더 나은 성능(정확도)을 기록했으나, MLA 적용 전후의 통제된 ablation이 아니며, 일부 벤치마크에서는 근소하게 낮은 성능을 보인다.  
  *근거*: [1, p.15][1, p.21]  
  *반대 근거*: ablation 결과 미제시, 일부 벤치마크에서 낮은 성능.

- **도입 용이성(D4)**:  
  기존 Transformer 기반 LLM에 SW 레벨에서 적용 가능하며, 논문·구현 예시가 있다. 일부 자료에서 drop-in replacement 가능성을 언급하나, 대규모 상용 서빙 스택에서의 완전 호환성·운영 리스크에 대한 실증적 검증은 부족하다.  
  *근거*: [1, p.6][1, p.29][3][19]  
  *반대 근거*: 대규모 운영 환경에서의 실증적 사례 부족.

#### HW: ITME (CXL-Hybrid 계층 메모리 확장)

- **비용 효율(D1)**:  
  CXL-hybrid memory로 GPU 서버의 메모리 용량을 TB 단위로 확장, throughput 최대 35.7% 개선이 보고된다. 다만, 실제 비용 절감 수치·TCO 분석은 제공되지 않고, 하드웨어 추가가 필요하다.  
  *근거*: [2, p.1][2, p.4][2, p.6][2, p.11]  
  *반대 근거*: 추가 HW 필요, 비용 절감 수치 미제공.

- **처리량·지연(D2)**:  
  multi-tier prefetching으로 throughput 개선, 초기 turn에서는 CPU-offload baseline과 유사한 지연을 보이나, turn 증가·I/O contention 시 성능 하락, local GPU memory 대비 latency 한계가 있다.  
  *근거*: [2, p.9][2, p.10][2, p.11]  
  *반대 근거*: turn 10–20에서 성능 하락, latency 한계.

- **품질 유지(D3)**:  
  정확도·출력 품질에 대한 벤치마크 결과가 제공되지 않아 판단 유보.  
  *근거*: 없음  
  *반대 근거*: [2, p.9]

- **도입 용이성(D4)**:  
  vLLM 기반 SW 스택, user-level prefetcher API 연동, CXL-hybrid memory, PCIe Gen5 SSD, RDMA 지원 RNIC 등 HW 추가가 필요하다. 기존 서빙 스택과의 호환성은 vLLM 기반에서 확인되나, 완전한 SW 변경만으로 도입되는 것은 아니며, 일부 커널/시스템 수정이 필요하다.  
  *근거*: [2, p.2][2, p.5][20][21][15]  
  *반대 근거*: HW 추가·시스템 수정 필요.

---

## 5. 시사점

관점별 평가가 엇갈리는 주요 지점은 다음과 같다.

| 주제 | 대상 | 시장 관점 | 도메인 관점 | 엇갈리는 이유 |
|---|---|---|---|---|
| 시장 성장성 인식 | HW | CXL 기반 메모리 확장 시장의 높은 성장성(29% 이상 CAGR), AI/LLM 추론에서 TB급 수요 급증 [13][14][22][23] | ITME는 실제 비용 절감·품질 유지·운영 복잡성 등 실질적 도입 장벽이 강조됨 [2, p.1][2, p.4][2, p.6][2, p.11] | 시장성은 전체 시장 성장에 주목, 도메인 적용은 실질 도입 장벽에 주목 |
| 기술 도입 용이성/확장성 | SW | MLA는 논문·오픈소스·프레임워크 지원 등 확장성 긍정 평가 [8][9][10][11][12] | 대규모 운영 환경에서의 호환성·운영 리스크 등 실증적 검증 부족 [1, p.6][1, p.29][3][19] | 시장성은 적용 모델·프레임워크 다양성, 도메인 적용은 실제 도입 난이도에 주목 |
| 품질 및 성능 검증 | 공통 | 저자 기준 성능 개선 수치(처리량, 비용 등) 제시, 독립적 대규모 서비스 도입 근거 부족 | MLA는 품질 손실·latency(SLO) 유지 검증 부족, ITME는 정확도·출력 품질 데이터 미제공 | 시장성은 확산·성장성, 도메인 적용은 실증적 품질·성능 데이터 중시 |

기술 간 인식 차이는 SW(MLA)가 구조 변경만으로 캐시 오버헤드를 줄이고, HW(ITME)는 물리적 메모리 공간을 확장한다는 점에서 출발한다. MLA는 프레임워크 지원·모델 적용의 확장성이 강조되나, 실제 대규모 운영 환경에서의 호환성·운영 리스크에 대한 실증적 데이터가 부족하다. ITME는 시장 성장성·수요 측면에서 긍정적으로 인식되나, 실제 도입에는 HW 추가·운영 복잡성·품질 검증 등 장벽이 존재한다.

두 기술은 각각 데이터(캐시) 크기를 줄이는 SW 최적화와, 물리적 메모리 공간을 넓히는 HW 확장으로 병행 적용이 가능하다. 실제로 vLLM 등 일부 LLM 서빙 프레임워크에서 SW/HW 병행 최적화 연구가 진행되고 있으나, 두 논문이 결합 효과를 직접 실험한 근거는 없으므로 이는 검증되지 않은 가능성에 머문다. [1, p.6][2, p.2][15]

공통적으로, 저자 자체 실험 및 간접 시장 자료에 의존하고, 제3자 독립 벤치마크 및 대규모 상용 서비스 도입 근거가 부족하다는 점에서 한계가 지적된다.

---

## 6. 한계점

본 보고서는 공개 논문 및 웹 자료 기반으로 작성되었으며, 다음과 같은 한계가 존재한다.

- **저자 자체 실험 의존**: DeepSeek-V2 MLA와 ITME 모두 저자 보고 기준의 성능·효율 수치에 의존하며, 제3자 독립 벤치마크 및 대규모 상용 서비스 도입 근거가 부족하다. [1, p.1][2, p.11]
- **시장 자료의 간접성**: 시장성 평가에서 인용된 시장 규모·성장률 수치는 각 기관별로 범위·정의가 상이하며, 기술별 직접적 시장 규모와는 차이가 있을 수 있다. [4][13][14][5]
- **판단 유보 항목**: ITME의 품질 유지(D3) 등 일부 평가는 근거 부족으로 '판단 유보' 등급을 부여했다.
- **실제 운영 환경 데이터 부족**: MLA의 품질 손실, ITME의 정확도·출력 품질 등 핵심 성능 지표에 대한 실증적 데이터가 부족하다.
- **상용화·생태계 확장 단계 미도달**: ITME는 FPGA 기반 프로토타입 수준으로, 실제 대규모 상용 운영 환경에서의 성능·비용 효과 데이터가 부족하다.
- **SW/HW 결합 효과 미검증**: 두 논문 모두 SW/HW 병행 적용의 직접적 실험·벤치마크를 제공하지 않는다.
- **확증편향 방지 조치**:  
  1. 질의 쌍 구성(성과/한계),  
  2. 반대 근거 강제,  
  3. 근거 출처 구분(저자 보고/웹/간접),  
  4. 판단 유보 등급 도입,  
  5. 생성·판정 모델 분리,  
  6. 이중 점검(금칙어·Judge),  
  7. Reflection(논문 원문·수치 대조)  
  등 7가지 조치를 적용해 편향을 최소화했다.

---

## REFERENCE

### 논문

[1] DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434.

[2] Jang, H. et al.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556.

[3] Ji, T. et al.(2025). Towards Economical Inference: Enabling DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs. *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 33313-33328.

### 기타

[4] Dataintelo(2025-06-28). *Cross-Platform LLM Inference Engine Market Research Report 2034*. dataintelo.com, https://dataintelo.com/report/cross-platform-llm-inference-engine-market

[5] Global Market Insights Inc.(2025-09-01). *Enterprise LLM Market Size & Share, Growth Analysis 2034*. Global Market Insights Inc., https://www.gminsights.com/industry-analysis/enterprise-llm-market

[6] industryresearch.biz(n.d.). *Large Language Model(LLM) Market - Size, Share & Industry Forecast*. industryresearch.biz, https://www.industryresearch.biz/market-reports/large-language-model-llm-market-102241

[7] DeepWiki(2026-01-24). *Multi-head Latent Attention (MLA) | deepseek-ai/DeepSeek-V3*. DeepWiki, https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-(mla)

[8] pub.towardsai.net(n.d.). *vLLM, SGLang, TRT-LLM: LLM Engine Guide*. pub.towardsai.net, https://pub.towardsai.net/part-3-implementation-engine-level-choosing-the-runtime-that-gives-you-these-for-free-b0e9081205b0

[9] SGLang(n.d.). *DeepSeek Models*. SGLang, https://sgl-project-sglang-93.mintlify.app/models/deepseek

[10] DEV Community(2026-05-20). *DeepSeek-V3: The 671B MoE Model You Can Run Locally in 2026*. DEV Community, https://dev.to/rams901/deepseek-v3-the-671b-moe-model-you-can-run-locally-in-2026-30o4

[11] LLMS3.com(n.d.). *Multi-Head Latent Attention (MLA) | LLMS3*. LLMS3.com, https://llms3.com/node/multi-head-latent-attention-mla

[12] Sebastian Raschka(n.d.). *Multi-Head Latent Attention (MLA)*. Sebastian Raschka, PhD, https://sebastianraschka.com/llms-from-scratch/ch04/05_mla

[13] Dataintelo(2025-09-30). *CXL Memory Module Market Research Report 2034*. dataintelo.com, https://dataintelo.com/report/cxl-memory-module-market

[14] Spherical Insights(n.d.). *CXL Memory Expansion Market Growth, Forecast Report 2035*. Spherical Insights, https://www.sphericalinsights.com/reports/cxl-memory-expansion-market

[15] medium.com(n.d.). *How We Solved the KV Cache Bottleneck in LLM Inference*. medium.com, https://medium.com/@jooho.lee_93960/how-we-solved-the-kv-cache-bottleneck-in-llm-inference-with-cxl-shared-memory-d5133dcfb044

[16] Sandeep Dattaprasad(2025-11-20). *How CXL Transforms RAG and KV Cache Performance*. ASTERA LABS, INC., https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance

[17] Introl(2026-04-27). *CXL 4.0 Infrastructure Planning Guide | Introl Blog*. Introl, https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-ai-memory-pooling-2025

[18] Samsung Semiconductor Global(2026-09-04). *Breaking AI Memory Limits with CXL Memory Pooling*. Samsung Semiconductor Global, https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling

[19] GitHub(n.d.). *junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent*. GitHub, https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention

[20] mlforsystems.org(n.d.). *Exploring CXL-based KV Cache Storage for LLM Serving Yupeng Tang1∗*. mlforsystems.org, https://mlforsystems.org/assets/papers/neurips2024/paper17.pdf

[21] Amit Golander(2026-03-13). *How CXL Memory Expansion Improves AI Economics*. ASTERA LABS, INC., https://www.asteralabs.com/resources/blog/inference-tokenomics-how-cxl-memory-expansion-improves-ai-economics

[22] fortunebusinessinsights.com(n.d.). *CXL Memory Pooling Appliance Market Size, Share [2026*. fortunebusinessinsights.com, https://www.fortunebusinessinsights.com/cxl-memory-pooling-appliance-market-117960

[23] snia.org(n.d.). *The Rebirth of Persistent Memory: How CXL lays*. snia.org, https://www.snia.org/blog/2026/rebirth-persistent-memory-how-cxl-lays-foundation-current-and-next-generation-persistent
