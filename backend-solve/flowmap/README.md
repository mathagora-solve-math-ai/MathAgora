# Flow Map Generator

LLM 풀이를 분석하여 Step별 Flow Map을 자동 생성하는 시스템

## 구조

```
flowmap/
├── schemas.py           # Input/Output 데이터 스키마
├── generator.py         # Flow Map Generator (LLM agent)
├── test_generator.py    # 테스트 스크립트
├── visualize_flowmap.ipynb  # Jupyter 시각화
└── outputs/
    ├── flowmap_prob1.json   # 7번 문제 Flow Map
    ├── flowmap_prob22.json  # 22번 문제 Flow Map
    ├── flowmap_prob1.png    # 시각화 이미지
    └── flowmap_prob22.png
```

## Flow Map이란?

여러 LLM의 step-by-step 풀이를 분석하여:
1. **유사한 풀이 단계끼리 그룹핑**
2. **각 그룹에 이름 작명** (예: "도함수 구하기")
3. **모델별 step 흐름 연결** (flow chart)

### 시각화 예시

```
      GPT-5.2          Claude Opus      Gemini 3 Pro
        │                  │                 │
  ┌─────┴──────┬───────────┴────┬────────────┴─────┐
  │ 도함수 구하기 │  도함수 구하기  │  도함수 구하기    │  ← Group 0
  └─────┬──────┴───────────┬────┴────────────┬─────┘
        │                  │                 │
  ┌─────┴──────┬───────────┴────┬────────────┴─────┐
  │ 임계점 찾기  │   임계점 찾기   │ 인수분해 및 0...  │  ← Group 1
  └─────┬──────┴───────────┬────┴────────────┬─────┘
        │                  │                 │
       ...                ...               ...
```

## 사용법

### 1. Flow Map 생성

```bash
cd flowmap
python3 test_generator.py
```

prob1(7번), prob22(22번) 두 문제에 대해 Flow Map JSON 생성

### 2. 시각화 (Jupyter Notebook)

```bash
jupyter notebook visualize_flowmap.ipynb
```

또는 VS Code에서 ipynb 파일 직접 열기

## Output 스키마

```json
{
  "groups": [
    {
      "group_id": 0,
      "group_name": "도함수 구하기",
      "steps": [
        {
          "model": "gpt-5.2",
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "f(x)=1/3x³-2x²-12x+4 이므로\nf'(x)=x²-4x-12"
        },
        {
          "model": "claude-opus-4.5",
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "극대와 극소를 찾기 위해 f(x)를 미분합니다.\n..."
        }
      ]
    }
  ],
  "flows": [
    {"model": "gpt-5.2", "from_step": 0, "to_step": 1},
    {"model": "gpt-5.2", "from_step": 1, "to_step": 2},
    {"model": "claude-opus-4.5", "from_step": 0, "to_step": 1}
  ]
}
```

## 실험 결과

### Prob1 (7번, 3점 - 쉬운 문제)
- **5개 그룹**, 완벽한 1:1:1 정렬
- 모든 그룹에 3개 모델 전부 참여
- 깔끔한 flow chart

### Prob22 (22번, 4점 - 어려운 문제)
- **8개 그룹**, 복잡한 구조
- Gemini는 파싱 실패로 미참여
- GPT(9 step) vs Claude(8 step) 구조 차이
- 일부 그룹에 같은 모델의 여러 step 포함

## 핵심 발견

1. **그룹핑 품질**: 쉬운 문제에서 LLM agent가 완벽히 정렬
2. **작명 품질**: "도함수 구하기", "극대·극소 판별" 등 직관적 이름
3. **어려운 문제 한계**: 모델 간 풀이 전략 차이로 그룹 복잡도 증가

## 다음 단계

- [ ] 프론트엔드와 연동
- [ ] 실시간 스트리밍 중 부분 flow map 생성
- [ ] 그룹 이름 품질 개선 (더 간결하게)
- [ ] 분기/병합 지점 시각적 강조
