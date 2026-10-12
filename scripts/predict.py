"""
학습된 weight로 새 이미지에 container_seal 검출(inference)을 실행하고 결과를 저장한다.

- 이 스크립트는 예측 결과를 만들기만 한다. 학습, metric 계산, 모델 선택은 하지 않는다.
- 원본 이미지는 읽기만 하고 수정하지 않는다.
- held-out test 폴더는 source로 받지 않는다 (test는 evaluate.py로 이미 한 번 평가함).

사용법 (프로젝트 루트에서):
    python scripts/predict.py --weights outputs/runs/yolo11n_seed1/weights/best.pt --source path/to/image.jpg
    python scripts/predict.py --weights outputs/runs/yolo11n_seed1/weights/best.pt --source path/to/folder --device cpu

출력:
    outputs/predictions/<run_name>/            box가 그려진 예측 이미지
    outputs/predictions/<run_name>/labels/     검출 결과 .txt (class x_center y_center width height confidence)
    (weight가 outputs/smoke/ 아래에 있으면 outputs/smoke/predictions/ 아래에 저장)
"""

import argparse
import sys
from pathlib import Path

from common import OUTPUTS_DIR, SPLIT_INFO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOP_K = 5  # 이미지별로 출력할 confidence 개수 (높은 순)
HELD_OUT_TEST_DIR = SPLIT_INFO.parent / "test"  # data/processed/container_seal_v2/test


def collect_images(source):
    """단일 이미지 파일 또는 폴더 안의 이미지 목록을 돌려준다 (하위 폴더는 보지 않음)."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            sys.exit(f"[오류] 이미지 파일이 아닙니다 (지원: {', '.join(sorted(IMAGE_EXTS))}): {source}")
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            sys.exit(f"[오류] 폴더에 이미지가 없습니다: {source}")
        return images
    sys.exit(f"[오류] source를 찾을 수 없습니다: {source}")


def default_run_name(weights, source):
    """출력 폴더 이름: <학습 run 이름>_<source 이름>. 예: yolo11n_seed1_image001"""
    run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    return f"{run}_{source.stem if source.is_file() else source.name}"


def main():
    parser = argparse.ArgumentParser(description="container_seal 검출 inference (예측 결과 저장만 함)")
    parser.add_argument("--weights", required=True, help="사용할 best.pt 경로")
    parser.add_argument("--source", required=True, help="이미지 파일 1개 또는 이미지가 들어 있는 폴더")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 입력 크기 (기본 640, 학습 조건과 동일)")
    parser.add_argument("--conf", type=float, default=0.25, help="이 값보다 낮은 confidence의 box는 버림 (기본 0.25)")
    parser.add_argument("--device", default=None, help="'0'(첫 번째 GPU) 또는 'cpu'. 생략하면 GPU가 있으면 GPU, 없으면 CPU")
    parser.add_argument("--name", default=None, help="출력 폴더 이름 (생략하면 <run>_<source 이름>)")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    if not weights.exists():
        sys.exit(f"[오류] weight 파일이 없습니다: {weights}")
    if not 0 <= args.conf <= 1:
        sys.exit("[오류] --conf 는 0~1 사이 값이어야 합니다.")

    # held-out test 이미지는 예측 대상으로 받지 않는다 (test 재평가 용도로 쓰지 않기 위함)
    if source == HELD_OUT_TEST_DIR.resolve() or source.is_relative_to(HELD_OUT_TEST_DIR.resolve()):
        sys.exit(f"[오류] held-out test 데이터는 예측 대상으로 사용하지 않습니다: {source}")

    images = collect_images(source)

    # smoke test weight이면 결과도 outputs/smoke/ 아래에 저장한다 (train.py / evaluate.py와 같은 규칙)
    smoke_dir = OUTPUTS_DIR / "smoke"
    base_dir = smoke_dir if weights.is_relative_to(smoke_dir.resolve()) else OUTPUTS_DIR

    run_name = args.name or default_run_name(weights, source)
    save_dir = base_dir / "predictions" / run_name
    # 같은 이름의 결과가 있으면 덮어쓰거나 이름을 바꾸지 않고 중단한다
    if save_dir.exists():
        sys.exit(f"[오류] {save_dir} 이 이미 있습니다. --name 으로 다른 이름을 주거나 기존 폴더를 직접 옮기세요.")

    # Ultralytics는 import가 느리므로 인자 검사가 끝난 뒤에 불러온다
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # stream=True: 이미지를 한 장씩 처리해 폴더가 커도 메모리를 적게 쓴다
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,  # box를 그린 예측 이미지 저장 (원본과 별도 파일)
        save_txt=True,  # 검출 결과를 YOLO 형식 .txt로 저장 (검출이 없는 이미지는 파일 없음)
        save_conf=True,  # .txt에 confidence도 함께 기록
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=False,
        stream=True,
        verbose=False,
    )

    print(f"weights : {weights}")
    print(f"source  : {source} ({len(images)}장)")
    print(f"설정    : imgsz={args.imgsz}, conf={args.conf}, device={args.device or 'auto'}\n")
    total = 0
    for result in results:
        n = len(result.boxes)
        total += n
        line = f"  {Path(result.path).name}: {n} detections"
        if n:
            # 전부 출력하면 길어지므로 confidence가 높은 순서로 최대 TOP_K개만 보여준다
            top = sorted(result.boxes.conf.tolist(), reverse=True)[:TOP_K]
            line += f", top confidences=[{', '.join(f'{c:.2f}' for c in top)}]"
        print(line)

    print(f"\n[완료] 이미지 {len(images)}장, 검출 {total}개")
    print(f"결과 저장: {save_dir}")


if __name__ == "__main__":
    main()
