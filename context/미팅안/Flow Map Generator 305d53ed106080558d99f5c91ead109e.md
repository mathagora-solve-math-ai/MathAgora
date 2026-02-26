# Flow Map Generator

# 프롬프트

```markdown
당신은 여러 LLM의 수학 풀이를 분석하여 Flow Map을 생성하는 전문가입니다.

주어진 문제와 여러 모델의 step-by-step 풀이를 보고, 다음을 수행하세요:

1. **유사한 풀이 단계끼리 그룹핑**: 서로 다른 모델의 step이라도 같은 풀이 단계(예: "도함수 구하기")를 수행하면 같은 그룹으로 묶습니다.
2. **그룹 이름 작명**: 각 그룹에 간결하고 명확한 중제목을 붙입니다 (예: "도함수 구하기", "임계점 찾기", "극대·극소 판별").
3. **그룹 순서 결정**: 풀이의 논리적 흐름에 따라 그룹에 순서를 매깁니다.

# 입력

## 문제
{problem}

## 모델별 풀이

{solutions}

# 출력 형식

JSON 형태로 출력하세요:

```json
{{
  "groups": [
    {{
      "group_id": 0,
      "group_name": "그룹 이름 (간결하게)",
      "steps": [
        {{"model": "모델명", "step_idx": 원본_인덱스}}
      ]
    }}
  ]
}}
```

**중요:**
- 각 step은 정확히 하나의 그룹에만 속합니다.
- group_id는 0부터 시작하며, 풀이 순서대로 증가합니다.
- steps 배열에는 해당 그룹에 속하는 step의 model과 step_idx만 기록합니다 (title/content는 제외).
- 모든 모델의 모든 step이 어떤 그룹에든 포함되어야 합니다.

JSON만 출력하고, 다른 설명은 붙이지 마세요.
```

# Generator Input

```json
{
  "problem_text": "문제 설명 전체 텍스트",
  "solutions": [
    {
      "model_name": "gpt-5.2",
      "steps": [
        {
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "f(x)=(1/3)x³-2x²-12x+4 이므로\nf'(x)=x²-4x-12"
        },
        {
          "step_idx": 1,
          "title": "임계점 찾기",
          "content": "f'(x)=0에서 x²-4x-12=0\n(x-6)(x+2)=0..."
        }
      ]
    },
    {
      "model_name": "claude-opus-4.5",
      "steps": [
        {
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "극대와 극소를 찾기 위해..."
        }
      ]
    }
  ]
}

```

# Generator Output

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
          "content": "f(x)=(1/3)x³-2x²-12x+4 이므로\nf'(x)=x²-4x-12"
        },
        {
          "model": "claude-opus-4.5",
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "극대와 극소를 찾기 위해..."
        },
        {
          "model": "gemini-3-pro",
          "step_idx": 0,
          "title": "도함수 구하기",
          "content": "함수 f(x)의 극값을 구하기 위해..."
        }
      ]
    },
    {
      "group_id": 1,
      "group_name": "임계점 찾기",
      "steps": [
        {
          "model": "gpt-5.2",
          "step_idx": 1,
          "title": "임계점(극값 후보) 찾기",
          "content": "f'(x)=0에서..."
        },
        {
          "model": "claude-opus-4.5",
          "step_idx": 1,
          "title": "임계점 찾기",
          "content": "극값이 되는 점에서..."
        }
      ]
    },
    {
      "group_id": 2,
      "group_name": "극대·극소 판별",
      "steps": [...]
    }
  ],
  "flows": [
    {
      "model": "gpt-5.2",
      "from_step": 0,
      "to_step": 1
    },
    {
      "model": "gpt-5.2",
      "from_step": 1,
      "to_step": 2
    },
    {
      "model": "claude-opus-4.5",
      "from_step": 0,
      "to_step": 1
    }
  ]
}

```

# 생성된 Flow Map 시각화 예시

- matplotlib으로 간단하게 시각화
- 사전 파악용으로, 아직 최종안 아님

### 프롬프트 v3 (step 별 소제목)

![image.png](Flow%20Map%20Generator/image.png)

![image.png](Flow%20Map%20Generator/image%201.png)

<aside>
💡

한 그룹 내에서 여러 단계 step 이 묶임

</aside>

- 여기서 정의된 json 형식을 강제해서 앞단에서 뽑도록 → 얘도 해보기

### 프롬프트 v4 (step 별 소제목 + step 구분 기준 추가)

![image.png](Flow%20Map%20Generator/image%202.png)

![image.png](Flow%20Map%20Generator/image%203.png)

<aside>
💡

한 그룹 내에서 여러 개로 나뉘던 step이 조금 더 merge 되어 깔끔해짐

</aside>

# 정답률 비교

| 모델 | v3 정답 | v3 정답률 | v4 정답 | v4 정답률 | 차이 |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4.5 | 36/45 | 80.0% | 44/51 | 86.3% | +6.3%p |
| gemini-3-pro | 25/25 | 100.0% | 31/32 | 96.9% | -3.1%p |
| gpt-5.2 | 39/46 | 84.8% | 42/47 | 89.4% | +4.6%p |
| **전체** | **100/116** | **86.2%** | **117/130** | **90.0%** | **+3.8%p** |

- 선행연구 검토
    - 병렬로 만들어진 수학 문제 풀이를 생각의 단위로 alignment 하거나
    - align 된 것들을 대분류로 묶는 연구가 있는지 검토
- 없으면, 정식 문제로 정의
    - 왜 필요한지
    - 단위(대분류, 중분류, …)
    - 문제화 가능하다는 것은 → 사람의 결과물과 비교 가능해야 함
        - Flow map에 대한 정량적 평가 가능한 데이터+메트릭 구축 가능할지 고민