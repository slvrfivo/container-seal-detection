"""train.py / evaluate.py가 함께 쓰는 경로와 실행 환경 기록 함수."""

import csv
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
SPLIT_INFO = PROJECT_ROOT / "data" / "processed" / "container_seal_v2" / ".split_info.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"  # COCO pretrained weight 다운로드 위치 (*.pt는 Git 제외)


def git_commit():
    """현재 코드 버전(commit hash). 커밋 안 된 변경이 있으면 '-dirty'를 붙인다."""
    try:
        def run(*args):
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()

        commit = run("rev-parse", "--short", "HEAD")
        return commit + ("-dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def split_hash():
    """어떤 데이터 분할로 실험했는지 구분하기 위한 .split_info.json 해시 (앞 12자리)."""
    if not SPLIT_INFO.exists():
        return "missing"
    return hashlib.sha256(SPLIT_INFO.read_bytes()).hexdigest()[:12]


def device_name(device):
    """실험 기록용 장치 이름. GPU면 모델명(예: Tesla T4), CPU면 CPU 정보."""
    import torch

    if str(device).lower() != "cpu" and torch.cuda.is_available():
        index = int(str(device).split(",")[0]) if str(device).split(",")[0].isdigit() else 0
        return torch.cuda.get_device_name(index)
    try:
        # Ultralytics 로그와 같은 형태 (예: "Intel Core i5-8265U 1.60GHz")
        from ultralytics.utils.torch_utils import get_cpu_info

        cpu = get_cpu_info()
    except Exception:
        cpu = ""
    return f"CPU ({cpu or platform.processor() or platform.machine()})"


def environment_info(device):
    """실험 재현에 필요한 환경 정보."""
    import torch
    import ultralytics

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "split_hash": split_hash(),
        "device": str(device),
        "device_name": device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


def metrics_info(metrics):
    """Ultralytics val 결과에서 기록할 지표만 꺼낸다. (단일 클래스이므로 전체 평균 = container_seal 값)"""
    box = metrics.box
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "inference_ms_per_img": round(float(metrics.speed["inference"]), 2),
    }


def append_csv(path, row):
    """CSV에 한 줄 추가. 파일이 없으면 헤더부터 만든다. 열 구성이 바뀌었으면 기존 기록 보호를 위해 중단."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            raise SystemExit(
                f"[오류] {path} 의 열 구성이 현재 코드와 다릅니다. 기존 파일을 백업/이름 변경 후 다시 실행하세요."
            )
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
