# Container Seal Detection

컨테이너 seal을 검출하는 1-class object detection 프로젝트입니다.

> **현재 상태: 데이터 준비 및 분할 완료, 모델 학습 전 단계**
> 아직 모델을 학습하지 않았으므로 성능 수치는 없습니다.

## 데이터셋

| 항목 | 내용 |
|---|---|
| 라벨링 | Roboflow (Auto Label + 수작업 검수), YOLO11 형식으로 export |
| 구성 | 한 번의 연속 촬영에서 추출한 프레임 105장 (frame 1–106, frame 101 없음) |
| 해상도 | 1920×1080 |
| 클래스 | 1개 (`container_seal`) |
| bounding box | 118개 |

데이터(이미지, 라벨, ZIP)는 저장소에 포함하지 않습니다.

## 라벨링 과정

1. Roboflow의 Auto Label 기능으로 초기 bounding box를 생성했습니다.
2. 전체 이미지의 annotation을 직접 검수했습니다.
3. 검수하면서 잘못된 box를 고치고, 빠진 annotation을 추가했습니다.
4. YOLO11 형식으로 export했습니다 (v1).
5. v1 품질 점검(아래)에서 frame 28에 annotation이 빠진 것을 발견했고, 이를 수정해 v2를 만들었습니다.

## 데이터 품질 점검

1. **v1 export 점검**
   - 이미지와 라벨이 1:1로 대응하는지, YOLO 형식과 좌표 범위가 올바른지 확인했습니다. 오류는 없었습니다.
   - 빈 라벨 15장을 직접 확인해 아래처럼 분류했습니다.
     - 정상 negative: 12장
     - 라벨 누락 의심: 2장 (frame 13, 28)
     - 판단 보류: 1장 (frame 48)
2. **v2 export (라벨 수정)**
   - frame 28: 누락된 bounding box를 추가했습니다.
   - frame 13: seal이 손에 거의 가려져 bounding box 기준이 애매합니다. 그래서 라벨을 붙이지 않고 데이터셋에서 제외했습니다.
   - frame 48: 확인 결과 정상 negative로 유지했습니다.
   - v1과 비교해 라벨이 바뀐 것은 frame 28뿐이고, 이미지는 v1과 같습니다.

## 분할 전략

연속 촬영 프레임은 이웃한 프레임끼리 거의 같은 장면입니다. random split을 하면 train과 비슷한 이미지가 validation/test에 섞여 성능이 실제보다 높게 나옵니다. 그래서 **프레임 번호 구간 단위**로 나눴습니다.

- **test**: 촬영 구도가 바뀐 원거리 장면(frame 79–106) 전체를 held-out test로 고정했습니다.
- **validation**: 근접 장면(frame 1–76) 안에서 연속된 두 구간을 골랐습니다. 하나는 은색 seal이 나오는 25–30, 다른 하나는 파란 seal과 negative가 함께 있는 44–50입니다.
- **buffer**: train과 validation이 맞닿는 경계마다 1프레임씩 제외했습니다. 비슷한 프레임이 양쪽에 섞이는 것을 막기 위해서입니다.

| split | frame | 이미지 | bbox | empty label |
|---|---|---|---|---|
| train | 1–12, 14–23, 32–42, 52–76 | 58 | 46 | 12 |
| valid | 25–30, 44–50 | 13 | 12 | 1 |
| test | 79–106 (101 없음) | 27 | 54 | 0 |
| excluded | 13, 24, 31, 43, 51, 77, 78 | 7 | 6 | – |

제외 이유는 다음과 같습니다.

| frame | 이유 |
|---|---|
| 13 | bounding box 기준이 애매해 라벨을 붙이지 않음 |
| 24, 31, 43, 51 | validation 경계 buffer |
| 77 | 카메라 구도 전환 중인 프레임 |
| 78 | 이미 test와 같은 원거리 구도라, train에 넣으면 test 장면이 학습에 섞임 |

## 프로젝트 구조

```
container-seal-detection/
├── configs/
│   └── data.example.yaml    # 데이터셋 설정 예시 (실제 data.yaml은 스크립트가 생성)
├── data/
│   ├── raw/                 # Roboflow export ZIP (Git 제외)
│   └── processed/           # 분할된 데이터셋 (Git 제외)
├── outputs/                 # 학습·평가 결과 (Git 제외)
└── scripts/
    └── split_dataset.py     # 프레임 기준 분할 + 검증
```

## 재현 방법

1. Roboflow YOLO11 export ZIP을 `data/raw/robot-labeling.v2i.yolov11.zip`에 둡니다.
2. 분할 스크립트를 실행합니다. Python 표준 라이브러리만 사용합니다.

   ```bash
   python scripts/split_dataset.py
   ```

   - 원본 ZIP은 수정하지 않습니다. 결과는 `data/processed/container_seal_v2/`와 `configs/data.yaml`에 생성됩니다.
   - 실행 전에 아래 항목을 검증하고, 하나라도 맞지 않으면 파일을 만들지 않고 중단합니다.
     - 모든 프레임이 train/valid/test/excluded 중 정확히 한 곳에만 속하는지
     - 이미지와 라벨이 1:1로 대응하는지
     - YOLO 형식과 좌표 범위가 올바른지
   - 출력 폴더가 이미 있으면 덮어쓰지 않습니다. `--overwrite`를 붙이면 이 스크립트가 만든 출력만 교체합니다.

## 한계

- 모든 데이터가 한 번의 촬영 세션에서 나왔습니다. 다른 환경(조명, 컨테이너, 카메라)에서도 잘 동작할지는 이 데이터로 검증할 수 없습니다.
- validation이 13장(bbox 12개)으로 작아서, 평가 지표가 크게 흔들릴 수 있습니다.
- test bbox 54개 중 27개는 장면에 고정된 같은 seal입니다.

## 다음 단계

- [ ] YOLO 모델 학습 및 validation 평가
- [ ] held-out test 평가
