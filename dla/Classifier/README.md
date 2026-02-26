# CSAT vs SAT 시험지 분류기

이미지가 CSAT(수능)인지 SAT인지 분류하는 모듈. 휴리스틱(헤더 OCR + 키워드)과 CNN 방식을 지원합니다.

## 폴더 구조

- `classifier.py` – 분류 API (`classify`, `classify_batch`) 및 휴리스틱/CNN 로직
- `inference_classify.py` – 단일 이미지 추론 (CLI)
- `prepare_data.py` – train/val/test CSV 생성 (data/SAT, data/*_math_odd 수집)
- `train.py` – 2-class CNN 학습 (ResNet18/MobileNetV2)
- `classifier_data/` – prepare_data 실행 시 생성 (exam_classifier_*.csv)
- `classifier_checkpoints/` – train 실행 시 생성 (best.pt, last.pt)

## 사용법 (dla 폴더에서 실행)

```bash
cd dla

# 1) 데이터 목록 생성 (train/val/test CSV)
python -m Classifier.prepare_data --data_dir ../data
# CSAT에 전체 페이지 + crop 이미지 모두 포함 (기본). crop 제외: --no_include_crops
# split 균형 (이미지 단위 랜덤): python -m Classifier.prepare_data --data_dir ../data --split_mode random

# 2) CNN 학습 (선택, tqdm으로 진행률 표시)
python -m Classifier.train
# GPU 선택 (1,2,3번만 사용): python -m Classifier.train --gpus 1 2 3
# 멀티 GPU (DataParallel):     python -m Classifier.train --gpus 0 1 2 3
# 옵션: --epochs 20 --batch_size 32 --arch resnet18
# 예: python -m Classifier.train --epochs 30 --batch_size 16 --gpus 1 2 3
```

# 3) 단일 이미지 추론 (best.pt 있으면 CNN, 없으면 휴리스틱)
python -m Classifier.inference_classify /path/to/image.png
# 방법 지정: --method cnn | heuristic | hybrid
# 예: python -m Classifier.inference_classify /path/to/image.png --method cnn
```

## 파이프라인 연동

`run_pipeline_unified.py`에서 `Classifier` 패키지를 import하여 분류 후 SAT/CSAT별 parser를 실행합니다.

```bash
python run_pipeline_unified.py ../data/SAT/11-images --output_dir outputs_dla --method heuristic
```

## 환경 변수

- `EXAM_CLASSIFIER_MODEL`: CNN 체크포인트 경로 (classify 시 --cnn 미지정일 때 사용)
