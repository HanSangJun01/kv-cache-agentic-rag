# 실행 체크리스트 (범위 한정본)

> 범위 한정은 2가지뿐: ① 기술 선정은 (2안) Human 기반, Doc Pool에서 진영별 1개씩 직접 선택 ② 평가 관점은 2번 시장성 + 4번 도메인 적용만.
> 제출: **개발 산출물만** (설계 산출물 미대상), **2026-09-27(일) 자정까지 thread로 Git link(public) + PDF**. 상세는 `01-GUIDELINE-solo.md` §0, §9, §10.

## A. 선택/결정 (코드 짜기 전 확정)
- [ ] SW 진영 기술 1개 선택 (TurboQuant / DeepSeek-V2 MLA / KIVI) + **선정 사유**
- [ ] HW 진영 기술 1개 선택 (InfiniGen / ITME(CXL) / PIM·CXL) + **선정 사유**
- [ ] 도메인 1개 선택 (데이터센터·클라우드 / OnDevice AI / 장문맥 애플리케이션) — 원본상 선택 사항, 본 설계에서 1개로 고정
- [ ] 오픈소스 임베딩 모델 후보 3개 이상 → 최종 1개 + **선정 기준 서술**
      (※ "리더보드 상위"는 부적절 사유로 명시되어 있음)
- [ ] RAG 적용 에이전트 지정 (기술조사 / 시장평가 / 도메인평가 중 최소 1개)
- [ ] RAG 문서 세트 구성 — **총 200페이지 이내** (페이지 수 실제 집계해서 기록)

## B. 설계 (설계 산출물은 제출하지 않지만 README·코드에 반영 필수)
- [ ] State 스키마 표 작성 — 가이드라인 §7 표 기준 (관점별 결과는 **분리된 키**, `sources`/`trace` 는 reducer)
- [ ] Graph 흐름 mermaid 작성 — 이해관계자 노드 없음, 기술선정은 Human 입력, **검토 Loop + revise/approve/unverified Branch 포함**
- [ ] 에이전트별 입력·출력·도구 표 확정 (§4-1) — 시장 평가의 웹 검색 사용 명시
- [ ] 평가 기준 표(M1~M3, D1~D4) + 등급 루브릭 + 구조화 출력 스키마 확정 (§6)
- [ ] 종합 에이전트 출력 스키마(일치/상충/인식 차이/보완/유의점) 확정 (§6)
- [ ] Agentic RAG 루프 정의 (검색 → 관련성 판정 → 질의 재작성 1회 → 웹 fallback) + 루프 상한·recursion_limit (§5-2-1)
- [ ] 확증편향 방지 전략 7항목 정의 (§6) — 설계·코드·보고서 6장 세 곳에 동일하게
- [ ] 웹 질의는 "채택·성과" + "한계·제약(사실)" 쌍으로만 구성 — 반응·의견 수집 질의(criticism, review, opinion) 금지 (3번 관점 제외)
- [ ] Generator / Judge 모델 분리 결정 및 README Tech Stack 기재
- [ ] 보고서 목차 초안 (§8)
- [ ] 위 설계 표·그래프를 README `Agents` / `Architecture` 섹션에 넣을 초안 작성

## C. 개발
- [ ] `data/` RAG 문서 세트 적재 (PDF) — 선정 기술 논문 2편 + (선택) 시장 리포트·백서, 총 200p 이내
- [ ] RAG 파이프라인: 로딩 → 청킹 → 임베딩 → 벡터DB → 검색 → 컨텍스트 주입 (`tech` 메타데이터 필터 필수)
- [ ] Agentic RAG 루프 구현 + `trace` 로그 저장
- [ ] 검색 품질 측정: Hit Rate@K, MRR 수치 확보 (평가셋은 에이전트가 실제 던지는 질의 형태로, 모델 선정용과 분리)
- [ ] 에이전트 구현 (기술조사 / 시장평가 / 도메인평가 / 평가종합 / 보고서생성 / **보고서검토**) + 저장 노드
- [ ] 출처 레지스트리(`sources`) + 인용 태그 → REFERENCE 자동 생성
- [ ] 근거 검증(Reflection) + 수치-원문 정규식 대조 (§5-2-2)
- [ ] 보고서 검토 규칙 검사기 (목차 / SUMMARY 길이 / 10p / 금칙어 / 수치 근거) + LLM Judge
- [ ] LangGraph 그래프 조립 (시장·도메인 fan-out → 종합 fan-in → 보고서 ⇄ 검토 Loop → 저장), `outputs/graph.mmd` 저장
- [ ] 코드의 에이전트·State·그래프가 **README의 설계 표·mermaid와 일치**하는지 대조 (설계 구현 충실도 15점)
- [ ] `python app.py` 한 번으로 보고서가 실제 생성되는지 확인 (재현성 10점) — **검토 미통과여도 PDF는 반드시 생성**되고 상태가 표시되는지
- [ ] 자동 생성본이 검토를 통과하도록 프롬프트·검사기 튜닝 (사후 수동 정정본 제출은 재현성·보고서 배점 리스크)
- [ ] 디렉토리 구조 정리: data / agents / prompts / rag / outputs / app.py / README.md

## D. 산출물 제출 — 2026-09-27(일) 자정까지
- [ ] 평가 보고서 생성 — **10페이지 이내**, 맨 앞 SUMMARY(½p 이내), 맨 뒤 REFERENCE
- [ ] 보고서에 우열 판정/추천 문구가 없는지 검수
- [ ] 보고서 4장에 시장·도메인 관점만 있고, 이해관계자 의견(경쟁사 반응·개발자 평가·투자 시각)이 근거로 섞이지 않았는지 검수
- [ ] REFERENCE 표기 형식(논문/특허/웹) 준수, 실제 사용한 자료만
- [ ] PDF 파일명: `RAG-Output_{캠퍼스}_{X반}_{이름}.pdf`, 저장소 `outputs/`에도 포함
- [ ] README.md 작성 (Subject / Overview / Selected Technologies / Features / Tech Stack /
      Agents / Architecture / Directory Structure / Usage / Contributors — 개인별 역할, PM·PL 제외)
- [ ] README에 범위 한정 2가지(Human 기반 선정, 2·4번 관점만) 명시
- [ ] 새 환경에서 clone → 설치 → `python app.py` 재실행 확인 (재현성)
- [ ] GitHub 저장소 **public** 전환 확인
- [ ] thread에 **Git link + PDF** 제출

## E. 확인 필요
- [ ] 발표 여부 — 제출 지침에 언급 없음. 진행 시 README로 10분, 차별점 + 보고서 핵심 포인트 + Lessons Learned

## F. 범위 밖 (하지 말 것)
- [x] ~~에이전트 기반 기술 선정 (1안)~~
- [x] ~~Doc Pool 밖 기술 선정, 진영별 2개 이상 선정~~
- [x] ~~기술 성숙도(TRL) 관점~~
- [x] ~~이해관계자 관점 (및 이해관계자 평가 에이전트)~~
- [x] ~~설계 산출물 PDF (RAG-Design_...)~~
