# Flow Map 시각화 개선 (Sub-row 방식)

## 문제점

**같은 그룹에 같은 모델의 여러 step이 있을 때 겹치는 문제**

예: `2024_odd_common_14` (가장 복잡한 문제)
- Group 6: Claude Step 5, 6 → 같은 위치에 겹침 ❌
- Group 7: Claude Step 7, 8 → 같은 위치에 겹침 ❌
- Group 9: GPT Step 8, 9 → 같은 위치에 겹침 ❌

## 해결 방법: Sub-row 배치

### 핵심 아이디어

같은 그룹 내 같은 모델의 여러 step을 **세로로 분산 배치**

```
기존:
  Group 6 (row=6.0)
    Claude [Step 5, Step 6]  ← 모두 (claude, 6.0)에 겹침

개선:
  Group 6
    Claude Step 5 → (claude, 5.75)  ← 위로 offset
    Claude Step 6 → (claude, 6.25)  ← 아래로 offset
```

### 구현 로직

```python
# 1. 같은 그룹 내 각 모델의 step 개수 파악
model_steps_in_group = {}
for step in group['steps']:
    model = step['model']
    if model not in model_steps_in_group:
        model_steps_in_group[model] = []
    model_steps_in_group[model].append(step)

# 2. Sub-row offset 계산
model_steps = model_steps_in_group[model]
sub_idx = step들 중 몇 번째인지

if len(model_steps) > 1:
    # 여러 step이 있으면 세로로 분산 (-0.25 ~ +0.25)
    total_spread = 0.5
    offset = (sub_idx - (len(model_steps) - 1) / 2) * (total_spread / (len(model_steps) - 1))
else:
    offset = 0

actual_row = row + offset
```

### 추가 개선사항

1. **박스 크기 조정**
   - 여러 step이 있으면 박스 높이를 줄여서 겹치지 않게
   ```python
   box_height = 0.7 / len(model_steps)
   ```

2. **Step 번호 표시**
   - 여러 step이 있을 때 `[step_idx]` 형식으로 구분
   ```python
   if len(model_steps) > 1:
       title_text = f"[{step_idx}] {title_text}"
   ```

3. **폰트 크기 조정**
   - 작은 박스에는 작은 폰트 사용

## 결과

### Before (기존)
- ❌ Step들이 완전히 겹침
- ❌ 화살표가 같은 위치 → 같은 위치라 보이지 않음
- ❌ 어떤 step이 있는지 알 수 없음

### After (개선)
- ✅ 모든 step이 명확히 표시됨
- ✅ 화살표 흐름이 보임
- ✅ [step_idx] 번호로 구분 가능
- ✅ Sub-row offset으로 순서 파악 가능

## 사용 방법

### Jupyter Notebook
```bash
jupyter notebook visualize_v3_fixed.ipynb
```

### 테스트할 문제
- `2024_odd_common_14`: 가장 복잡 (10 groups, 3건의 중복)
- `2024_odd_common_1`: 간단한 예제

## 파일 위치

- **Fixed 노트북**: `flowmap/visualize_v3_fixed.ipynb`
- **기존 노트북**: `flowmap/visualize_v3.ipynb`
- **출력**: `outputs/v3_visualizations/*_fixed.png`

## 개선 효과

**2024_odd_common_14 기준:**
- Group 6, 7, 9에서 겹침 해결
- 총 18개 flows 중 6개가 안 보이던 문제 해결
- 모든 step의 제목과 순서 확인 가능
