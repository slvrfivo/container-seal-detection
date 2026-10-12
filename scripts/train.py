"""
YOLO11 학습 1회를 실행하고 validation 결과를 outputs/experiments.csv에 기록한다.

모델 비교를 공정하게 하기 위해 학습 조건(BASELINE)은 코드에 고정하고,
명령행에서는 모델 / seed / device만 바꿀 수 있다.

사용법 (프로젝트 루트에서):
    python scripts/train.py --model yolo11n --seed 0 --device 0      # Colab GPU 본 학습
    python scripts/train.py --model yolo11n --seed 0 --device cpu --smoke   # 로컬 동작 확인 (1 epoch)

출력:
    outputs/runs/<model>_seed<seed>/        Ultralytics 결과 (args.yaml, results.csv, weights/best.pt, 그래프)
    outputs/experiments.csv                  실행 1회 = 1줄 요약
    (--smoke이면 outputs/smoke/ 아래에 따로 저장되어 본 실험 기록과 섞이지 않음)
"""

import argparse
import csv
import sys
from pathlib import Path

from common import DATA_YAML, OUTPUTS_DIR, WEIGHTS_DIR, append_csv, environment_info, metrics_info

# ---------------------------------------------------------------------------
# 학습 조건 (확정한 baseline). 여기 값은 README에 그대로 옮겨 재현 조건으로 쓴다.
# 적지 않은 값(augmentation, optimizer=auto, learning rate 등)은 Ultralytics 기본값을 사용한다.
# ---------------------------------------------------------------------------
BASELINE = {
    "imgsz": 640,
    "epochs": 100,
    "batch": 8,
    "patience": 30,  # val mAP50-95가 30 epoch 동안 개선되지 않으면 조기 종료
    "deterministic": True,
    "workers": 2,
    "pretrained": True,  # COCO pretrained weight에서 시작
}
MODELS = ["yolo11n", "yolo11s"]
SMOKE_EPOCHS = 1


def find_best_epoch(results_csv):
    """Ultralytics results.csv에서 val mAP50-95가 가장 높았던 epoch과 실제 학습한 epoch 수를 찾는다."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    return int(float(best["epoch"])), len(rows)


def main():
    parser = argparse.ArgumentParser(description="YOLO11 container_seal 학습")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="0", help="'0'(첫 번째 GPU) 또는 'cpu'")
    parser.add_argument("--smoke", action="store_true", help=f"동작 확인용 {SMOKE_EPOCHS} epoch 실행 (본 실험 기록과 분리)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"[오류] {DATA_YAML} 이 없습니다. 먼저 python scripts/split_dataset.py 를 실행하세요.")

    # 본 실험과 smoke test의 출력 위치를 분리한다
    base_dir = OUTPUTS_DIR / "smoke" if args.smoke else OUTPUTS_DIR
    run_name = f"{args.model}_seed{args.seed}"
    run_dir = base_dir / "runs" / run_name
    log_csv = base_dir / "experiments.csv"

    # 같은 이름의 결과가 있으면 덮어쓰거나 이름을 바꾸지 않고 중단한다 (실험 기록 혼동 방지)
    if run_dir.exists():
        sys.exit(f"[오류] {run_dir} 이 이미 있습니다. 다시 학습하려면 기존 폴더를 직접 옮기거나 지우세요.")

    train_cfg = dict(BASELINE)
    if args.smoke:
        train_cfg["epochs"] = SMOKE_EPOCHS

    # Ultralytics는 import가 느리므로 인자 검사가 끝난 뒤에 불러온다
    from ultralytics import YOLO

    # COCO pretrained weight: weights/ 에 없으면 Ultralytics가 자동 다운로드
    WEIGHTS_DIR.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS_DIR / f"{args.model}.pt"))

    model.train(
        data=str(DATA_YAML),
        seed=args.seed,
        device=args.device,
        project=str(run_dir.parent),
        name=run_name,
        exist_ok=False,
        **train_cfg,
    )

    # 학습이 끝난 뒤 best.pt를 validation set으로 다시 평가해 기록한다 (test는 사용하지 않음)
    best_pt = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best_pt)).val(
        data=str(DATA_YAML),
        split="val",
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=args.device,
        project=str(run_dir),
        name="val_best",
        exist_ok=False,
    )
    best_epoch, epochs_ran = find_best_epoch(run_dir / "results.csv")

    row = {
        "run": run_name,
        "model": args.model,
        "seed": args.seed,
        **{k: train_cfg[k] for k in ("imgsz", "epochs", "batch", "patience")},
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        **metrics_info(metrics),
        **environment_info(args.device),
        "weights": best_pt.relative_to(OUTPUTS_DIR.parent).as_posix(),
    }
    append_csv(log_csv, row)

    print(f"\n[완료] {run_name}  (기록: {log_csv})")
    for key in ("epochs_ran", "best_epoch", "precision", "recall", "mAP50", "mAP50-95", "device_name"):
        print(f"  {key:<12} {row[key]}")


if __name__ == "__main__":  # Windows에서 dataloader worker가 정상 동작하려면 필요
    main()
