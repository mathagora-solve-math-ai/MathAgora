# Flow Map Generator - 현재 상태

> 2026-02-11

## ✅ 완료된 작업

### 1. Flow Map Generator 구현
- **Input/Output 스키마 정의** (`schemas.py`)
- **LLM Agent 기반 Generator** (`generator.py`)
  - Step 그룹핑 (유사한 단계끼리)
  - 그룹 이름 작명 (중제목)
  - Flow 연결 정보 생성

### 2. 검증 완료
- **prob1 (7번, 3점)**: 5개 그룹, 완벽한 정렬
- **prob22 (22번, 4점)**: 8개 그룹, 복잡한 구조

### 3. 전체 파이프라인 구축
- **46문제 자동 처리 파이프라인** (`pipeline_all_problems.py`)
- 진행 상황 모니터링 (`check_progress.sh`)
- 결과 분석 도구 (`analyze_results.py`)
- Jupyter 시각화 (`visualize_flowmap.ipynb`)

### 4. 현재 실행 중
**46문제 전체 Flow Map 생성** (백그라운드 실행 중)

```bash
# 진행 상황 확인
cd /Users/vusrhdns/acl2026demo/flowmap
./check_progress.sh

# 실시간 로그
tail -f outputs/all_problems/pipeline.log
```

예상 완료 시간: 30~60분

---

## 📊 검증된 결과

### 쉬운 문제 (7번)
```
5개 그룹 - 완벽한 정렬
├ Group 0: 도함수 구하기 (GPT, Claude, Gemini)
├ Group 1: 임계점 찾기 (GPT, Claude, Gemini)
├ Group 2: 극대·극소 판별 (GPT, Claude, Gemini)
├ Group 3: β - α 계산 (GPT, Claude, Gemini)
└ Group 4: 최종 답 (GPT, Claude, Gemini)
```

### 어려운 문제 (22번)
```
8개 그룹 - 복잡한 구조
├ Group 0: 조건 해석 및 정리 (GPT, Claude)
├ Group 1: 도함수 조건으로 a, b 관계식 도출 (GPT, Claude 2 steps)
├ Group 2: 정수 부호 조건을 만족하는 함수 형태 추론 (GPT, Claude 2 steps)
├ Group 3~7: ...
└ Group 7: 최종 답 (GPT, Claude)

* Gemini는 파싱 실패 (0 step)
```

---

## 📁 파일 구조

```
flowmap/
├── schemas.py                  # 데이터 스키마
├── generator.py                # Flow Map Generator
├── test_generator.py           # prob1, prob22 테스트
├── pipeline_all_problems.py    # 전체 문제 파이프라인
├── analyze_results.py          # 결과 분석
├── visualize_flowmap.ipynb     # Jupyter 시각화
├── check_progress.sh           # 진행 상황 확인
├── README.md                   # 사용법
└── outputs/
    ├── flowmap_prob1.json      # 7번 Flow Map
    ├── flowmap_prob22.json     # 22번 Flow Map
    └── all_problems/           # 46문제 전체 결과 (진행 중)
        ├── steps_*.json        # 각 문제의 step
        ├── flowmap_*.json      # 각 문제의 flow map
        ├── summary.json        # 전체 요약
        └── pipeline.log        # 실행 로그
```

---

## 🎯 핵심 성과

1. **LLM Agent 기반 자동화**
   - 수동 TF-IDF 대신 LLM이 직접 그룹핑 + 작명
   - 쉬운 문제: 완벽한 정렬
   - 어려운 문제: 복잡도를 잘 표현

2. **명확한 스키마 정의**
   - Input: 모델별 step list
   - Output: groups + flows
   - 프론트엔드 연동 준비 완료

3. **확장 가능한 파이프라인**
   - CSV에서 문제 로드
   - 여러 모델 동시 호출
   - Flow Map 자동 생성
   - 46문제 처리 중

---

## 🚀 다음 단계

### 완료 대기 중
- [ ] 46문제 파이프라인 완료 (진행 중, 30~60분 예상)
- [ ] 결과 분석 (`python3 analyze_results.py`)

### 후속 작업 (금요일까지)
- [ ] 프론트엔드 연동
  - Flow Map JSON 포맷 공유
  - 시각화 컴포넌트 구현 (은빈님)
- [ ] 실시간 스트리밍 지원 검토
- [ ] 그룹 이름 품질 개선 (필요시)

---

## 📝 교수님께 보고할 자료

1. **PoC 요약**: `poc/results/poc_summary_for_prof.md`
2. **Flow Map 예시**: `flowmap/outputs/flowmap_prob1.json`
3. **시각화**: Jupyter notebook 실행 결과
4. **전체 통계**: 46문제 완료 후 `analyze_results.py` 출력

---

## ⚡ 명령어 치트시트

```bash
# 진행 상황
cd /Users/vusrhdns/acl2026demo/flowmap
./check_progress.sh

# 로그 모니터링
tail -f outputs/all_problems/pipeline.log

# 완료 후 분석
python3 analyze_results.py

# 시각화
jupyter notebook visualize_flowmap.ipynb
```
