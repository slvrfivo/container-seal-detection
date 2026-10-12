"""
학습된 weight를 validation 또는 test set으로 평가한다.

- 기본은 validation. 모델 비교·선택에는 validation 결과만 사용한다.
- test는 최종 모델을 확정한 뒤 한 번만 평가한다. 실수를 막기 위해
  --split test 에는 --final 옵션이 반드시 필요하고, 이미 test 기록이 있으면 실행하지 않는다.

사용법 (프로젝트 루트에서):
    python scripts/evaluate.py --weights outputs/runs/yolo11n_seed0/weights/best.pt --device 0
    python scripts/evaluate.py --weights <최종 모델 best.pt> --split test --final --device 0

출력:
    outputs/evaluations/<split>_<run>/      Ultralytics 평가 결과 (그래프, 예측 예시)
    outputs/val_evaluation.csv / outputs/test_evaluation.csv   평가 1회 = 1줄 기록
    (weight가 outputs/smoke/ 아래에 있으면 위 결과도 outputs/smoke/ 아래에 저장되어 본 실험 기록과 섞이지 않음)
"""

import argparse
import sys
from pathlib import Path

from common import DATA_YAML, OUTPUTS_DIR, PROJECT_ROOT, append_csv, environment_info, metrics_info

IMGSZ = 640  # 학습 조건과 동일하게 유지
BATCH = 8


def main():
    parser = argparse.ArgumentParser(description="YOLO11 container_seal 평가")
    parser.add_argument("--weights", required=True, help="평가할 best.pt 경로")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="최종 모델 확정 후 test 평가임을 명시")
    parser.add_argument("--device", default="0", help="'0'(첫 번째 GPU) 또는 'cpu'")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        sys.exit(f"[오류] weight 파일이 없습니다: {weights}")
    if not DATA_YAML.exists():
        sys.exit(f"[오류] {DATA_YAML} 이 없습니다. 먼저 python scripts/split_dataset.py 를 실행하세요.")

    # smoke test weight이면 결과도 outputs/smoke/ 아래에 저장한다 (train.py --smoke와 같은 규칙)
    smoke_dir = OUTPUTS_DIR / "smoke"
    base_dir = smoke_dir if weights.is_relative_to(smoke_dir.resolve()) else OUTPUTS_DIR

    log_csv = base_dir / f"{args.split}_evaluation.csv"
    if args.split == "test":
        if not args.final:
            sys.exit("[오류] test 평가는 최종 모델 확정 후에만 합니다. 의도한 것이라면 --final 을 붙이세요.")
        if log_csv.exists():
            sys.exit(f"[오류] test 평가 기록이 이미 있습니다: {log_csv}\n       test는 한 번만 평가합니다.")

    # outputs/runs/<run>/weights/best.pt 구조라면 <run>을 이름으로 쓴다
    run_name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    eval_dir = base_dir / "evaluations" / f"{args.split}_{run_name}"
    if eval_dir.exists():
        sys.exit(f"[오류] {eval_dir} 이 이미 있습니다. 기존 결과를 직접 옮기거나 지운 뒤 다시 실행하세요.")

    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(DATA_YAML),
        split=args.split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=args.device,
        project=str(eval_dir.parent),
        name=eval_dir.name,
        exist_ok=False,
    )

    try:
        weights_label = weights.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        weights_label = str(weights)
    row = {
        "run": run_name,
        "split": args.split,
        **metrics_info(metrics),
        **environment_info(args.device),
        "weights": weights_label,
    }
    append_csv(log_csv, row)

    print(f"\n[완료] {args.split} 평가: {run_name}  (기록: {log_csv})")
    for key in ("precision", "recall", "mAP50", "mAP50-95", "inference_ms_per_img", "device_name"):
        print(f"  {key:<22} {row[key]}")


if __name__ == "__main__":
    main()
