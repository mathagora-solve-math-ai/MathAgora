# 선행연구 조사 프롬프트 (Claude Chat 앱용)

## 📋 기본 조사 프롬프트

```
당신은 AI/NLP/교육공학 분야 연구 조사 전문가입니다. 다음 주제에 대한 선행연구를 조사해주세요:

**주제**: 수학 문제 풀이의 단계별 alignment 및 대분류 방법론

**조사 범위**:
1. 병렬로 생성된 여러 수학 문제 풀이를 "생각의 단위"로 alignment 하는 연구
2. Aligned 된 풀이 단계들을 대분류/계층적으로 묶는 연구
3. Mathematical reasoning의 decomposition/annotation 연구
4. Solution tree, reasoning trace 비교/정렬 연구

**필요한 정보**:
- 논문 제목, 저자, 학회/저널, 년도
- 핵심 방법론 (어떻게 alignment/grouping 하는가?)
- 사용한 데이터셋
- 정량적 평가 메트릭 (있다면)
- 주요 발견/기여점

**검색 키워드 힌트**:
- "mathematical reasoning alignment"
- "solution step annotation"
- "reasoning trace comparison"
- "educational concept mapping"
- "step-by-step solution decomposition"
- "parallel solution alignment"

**출력 형식**:
각 연구를 다음 형태로 정리해주세요:

### [논문 제목] (년도)
- **저자**:
- **학회/저널**:
- **방법론**:
- **데이터셋**:
- **메트릭**:
- **기여점**:
- **관련도**: [1-5점, 우리 연구와의 관련성]
- **URL**:

마지막에 요약 섹션을 추가:
- 전체적인 연구 동향
- 우리 문제와 가장 유사한 연구 3개
- 아직 해결되지 않은 gap
```

---

## 🔍 심화 조사 프롬프트 (단계별)

### Phase 1: 광범위 탐색
```
**작업**: 수학 문제 풀이 alignment 관련 연구의 전체 지형도(landscape) 파악

**조사 영역**:
1. NLP/AI 학회 (ACL, EMNLP, ICLR, NeurIPS, ICML)
2. 교육공학 학회 (AIED, EDM, ITS)
3. 인지과학/학습과학 저널

**시간 범위**: 최근 5년 (2020-2025)

**검색 전략**:
- 먼저 survey/review 논문부터 찾기
- 각 survey의 reference 중요 논문 추적
- 최신 학회(2024-2025)의 관련 논문 확인

**질문**:
1. 이 분야에 대표적인 survey 논문이 있는가?
2. 가장 많이 인용되는 foundational work는?
3. 최근(2024-2025) 주목할 만한 연구는?
```

### Phase 2: 방법론 깊이 분석
```
**작업**: Alignment/Grouping 방법론의 기술적 세부사항 분석

**각 연구에 대해**:
1. **Input 형태**: 무엇을 align 하는가?
   - 자연어 풀이?
   - Symbolic expression?
   - 둘 다?

2. **Alignment 방법**:
   - Rule-based? ML-based? LLM-based?
   - Semantic similarity? Structural similarity?
   - Graph-based? Sequence-based?

3. **Grouping/Clustering 방법**:
   - Hierarchical? Flat?
   - Supervised? Unsupervised?
   - 대분류의 기준은?

4. **구현 가능성**:
   - 코드 공개 여부
   - 재현 가능성
   - 우리 데이터에 적용 가능성

**비교표 작성**:
논문명 | Alignment 방법 | Grouping 방법 | 코드 | 적용가능성
```

### Phase 3: 데이터셋 & 벤치마크 조사
```
**작업**: 수학 문제 풀이 관련 공개 데이터셋 및 벤치마크 조사

**필요 정보**:
1. 데이터셋 이름
2. 문제 수, 풀이 수
3. 언어 (영어/한국어/다국어)
4. Annotation 여부 (step, concept, etc.)
5. 다중 풀이(multiple solutions) 포함 여부
6. 라이센스 및 접근성

**주요 데이터셋 후보**:
- GSM8K, MATH, MathQA
- PRM800K (OpenAI)
- APPS, CodeContests
- 교육용 데이터셋 (ASSISTments, etc.)

**우리 데이터와 비교**:
- CSAT 수능 데이터의 독특한 점
- 기존 데이터셋으로 해결 안 되는 부분
```

### Phase 4: 평가 메트릭 조사
```
**작업**: Flow map 품질을 정량적으로 평가할 수 있는 메트릭 조사

**기존 연구의 메트릭**:
1. **Alignment 품질**:
   - Precision/Recall?
   - Agreement score (inter-annotator)?
   - Edit distance?

2. **Grouping 품질**:
   - Cluster purity?
   - Silhouette score?
   - Domain expert evaluation?

3. **Downstream task**:
   - 풀이 과정이 최종 정답 예측에 도움이 되는가?
   - Transfer learning 성능?

**새로운 메트릭 제안 가능성**:
- Flow map의 특성을 반영한 메트릭
- 사람의 풀이 과정과의 일치도
- 교육적 유용성
```

---

## 🎯 문제 정의 프롬프트 (선행연구가 없을 경우)

```
**상황**: 선행연구 조사 결과, 우리가 하려는 작업과 정확히 일치하는 연구가 없음

**작업**: 새로운 연구 문제로 정의하기

**답변해야 할 질문**:

1. **왜 이 문제가 필요한가?**
   - 기존 연구의 한계점
   - 우리 문제만의 독특한 점
   - 해결되면 어떤 가치가 있는가?
   - 누가 이 결과를 사용할 것인가?

2. **단위(granularity) 정의**:
   - 대분류: 어떤 기준으로?
   - 중분류: 어떤 기준으로?
   - 소분류(step): 어떤 기준으로?
   - 계층 구조의 명확한 정의

3. **문제의 형식화(formalization)**:
   - Input: 무엇이 주어지는가?
   - Output: 무엇을 생성해야 하는가?
   - Constraint: 어떤 제약이 있는가?
   - Objective: 최적화 목표는?

4. **사람과의 비교 가능성**:
   - 사람이 만든 flow map 수집 방법
   - 사람-AI 일치도 측정 방법
   - Ground truth 정의

5. **정량적 평가**:
   - 자동 평가 메트릭 설계
   - 사람 평가 프로토콜
   - Inter-annotator agreement 측정

**출력 형식**:
위 질문들에 대한 답을 구조화된 문서로 작성.
각 섹션은 연구 제안서(research proposal) 수준으로.
```

---

## 📊 체크리스트 & 옵션

### ✅ 조사 범위 옵션

**시간 범위**:
- [ ] 최근 2년 (2023-2025) - 최신 동향
- [ ] 최근 5년 (2020-2025) - 표준
- [ ] 최근 10년 (2015-2025) - 포괄적

**학회/저널 우선순위**:
- [ ] Tier 1: NeurIPS, ICML, ICLR, ACL, EMNLP
- [ ] Tier 2: AAAI, IJCAI, NAACL, AIED
- [ ] 저널: JMLR, TACL, Nature/Science AI
- [ ] 교육/인지과학: Cognition, Learning Sciences

**검색 전략**:
- [ ] arXiv 최신 논문
- [ ] Google Scholar 인용 추적
- [ ] Connected Papers 시각화
- [ ] OpenReview 토론 확인

### 🔧 조사 깊이 옵션

**Level 1 (빠른 조사, 1-2시간)**:
- 주요 키워드로 검색
- 최근 2년 상위 학회만
- 제목/초록만 읽기
- 10-20개 논문 리스트업

**Level 2 (표준 조사, 3-5시간)**:
- 여러 키워드 조합
- 최근 5년, Tier 1-2 학회
- Introduction + Conclusion 읽기
- 30-50개 논문 분석

**Level 3 (심층 조사, 1-2일)**:
- 체계적 문헌 조사 (systematic review)
- 인용 네트워크 추적
- 전체 논문 읽기
- Survey 논문 작성 수준

### 📝 출력 형식 옵션

**Option A: 요약 리스트**
- 논문당 3-5줄 요약
- 빠른 파악용

**Option B: 상세 분석**
- 논문당 1페이지 분석
- 방법론, 데이터, 메트릭 상세

**Option C: 비교표**
- Excel/Notion 표 형식
- 여러 논문 한눈에 비교

**Option D: 마인드맵**
- 연구들의 관계도
- 시각적 이해

---

## 🚀 실행 가이드

### Claude Chat 앱에서 사용하는 방법:

1. **기본 조사부터 시작**:
   ```
   위의 "기본 조사 프롬프트" 복사 → Claude에 붙여넣기
   ```

2. **결과 확인 후 심화**:
   ```
   Phase 1 프롬프트 실행 → 결과 분석 →
   Phase 2 프롬프트 실행 → ...
   ```

3. **옵션 설정**:
   ```
   "Level 2 조사로 진행해줘"
   "Option B 형식으로 출력해줘"
   "최근 3년만 봐줘"
   ```

4. **반복 개선**:
   ```
   "교육공학 쪽 연구를 더 찾아줘"
   "데이터셋 관련 부분을 더 상세히"
   "메트릭 부분만 따로 정리해줘"
   ```

### 추천 워크플로우:

```
Day 1 (3시간):
- 기본 조사 프롬프트 실행
- Phase 1 (광범위 탐색)
- 결과 정리 및 gap 파악

Day 2 (3시간):
- Phase 2 (방법론 분석)
- Phase 3 (데이터셋 조사)
- 비교표 작성

Day 3 (2시간):
- Phase 4 (메트릭 조사)
- 문제 정의 프롬프트 (필요시)
- 최종 리포트 작성
```

---

## 💡 Tips

1. **Claude Projects 활용**:
   - 새 프로젝트 생성: "Flow Map Literature Review"
   - 조사 결과를 프로젝트에 계속 추가
   - 컨텍스트 유지하며 심화 조사

2. **검색 키워드 확장**:
   - Claude에게 추가 키워드 제안 요청
   - 발견한 논문의 키워드 활용

3. **Citation Chaining**:
   - 좋은 논문 발견 시 references 추적
   - "이 논문이 인용한 중요 논문 3개를 더 조사해줘"

4. **비판적 평가**:
   - "이 방법론을 우리 데이터에 적용할 때 한계점은?"
   - "CSAT 데이터의 특성상 다르게 접근해야 할 부분은?"
