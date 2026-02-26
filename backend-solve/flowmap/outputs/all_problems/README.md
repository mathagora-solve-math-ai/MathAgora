# 전체 문제 Flow Map 생성 결과

## 현황

**Pipeline 실행 중**: 2024_math_odd.csv의 46문제 전체 처리

- Input: 2024 수능 수학 홀수형 46문제
- Models: GPT-5.2, Claude Opus 4.5, Gemini 3 Pro
- Prompt: v3_titled (step 제목 작명)

## 진행 상황 확인

```bash
cd /Users/vusrhdns/acl2026demo/flowmap

# 현재 진행 상황
./check_progress.sh

# 실시간 로그 모니터링
tail -f outputs/all_problems/pipeline.log
```

## 출력 파일

각 문제마다 2개 파일 생성:

```
outputs/all_problems/
├── steps_2024_odd_common_1.json      # Step 파싱 결과
├── flowmap_2024_odd_common_1.json    # Flow Map
├── steps_2024_odd_common_2.json
├── flowmap_2024_odd_common_2.json
├── ...
├── summary.json                       # 전체 요약
└── pipeline.log                       # 실행 로그
```

## 예상 소요 시간

- 문제당 약 30~60초 (LLM 호출 3회 + Flow Map 생성 1회)
- 전체 46문제: 약 30~60분 예상

## 완료 후 분석

Pipeline 완료 후:

```bash
python3 analyze_results.py
```

이 명령으로 다음 통계 확인 가능:
- 성공/실패 비율
- 문제 난이도별 평균 그룹 수
- 모델별 참여율
- 가장 복잡한/간단한 문제

## Step JSON 구조

```json
{
  "problem": {
    "prob_id": "2024_odd_common_1",
    "prob_area": "수학1",
    "prob_point": "2",
    "prob_desc": "문제 텍스트...",
    "answer": "정답"
  },
  "solutions": {
    "gpt-5.2": [
      {"title": "step 제목", "body": "step 내용"},
      ...
    ],
    "claude-opus-4.5": [...],
    "gemini-3-pro": [...]
  }
}
```

## Flow Map JSON 구조

```json
{
  "groups": [
    {
      "group_id": 0,
      "group_name": "그룹 이름",
      "steps": [
        {"model": "gpt-5.2", "step_idx": 0, "title": "...", "content": "..."},
        ...
      ]
    }
  ],
  "flows": [
    {"model": "gpt-5.2", "from_step": 0, "to_step": 1},
    ...
  ]
}
```

## 다음 단계

완료 후:
1. `analyze_results.py`로 통계 분석
2. 특정 문제 flow map 시각화 (Jupyter notebook)
3. 프론트엔드 연동용 데이터로 활용
