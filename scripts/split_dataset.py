"""
container_seal 데이터셋을 프레임 번호 기준으로 train / valid / test 로 나누는 스크립트.

- 입력: data/raw/robot-labeling.v2i.yolov11.zip (Roboflow YOLO11 export, 수정하지 않음)
- 출력: data/processed/container_seal_v2/{train,valid,test}/{images,labels}
        configs/data.yaml

연속 촬영 이미지이므로 random split 대신, 아래에 명시한 프레임 구간으로 나눈다.

사용법 (프로젝트 루트에서):
    python scripts/split_dataset.py
    python scripts/split_dataset.py --overwrite   # 이 스크립트가 만든 기존 출력만 교체
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로 설정 (이 파일 위치 기준으로 프로젝트 루트를 찾는다)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "robot-labeling.v2i.yolov11.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "container_seal_v2"
DATA_YAML_PATH = PROJECT_ROOT / "configs" / "data.yaml"

# 출력 폴더 안에 남겨두는 표시 파일. --overwrite 시 "이 스크립트가 만든 폴더"인지 확인하는 데 쓴다.
MARKER_FILE = ".split_info.json"

CLASS_NAMES = ["container_seal"]


def frames(*ranges):
    """(시작, 끝) 구간들을 프레임 번호 집합으로 바꾼다. 끝 번호도 포함한다."""
    result = set()
    for start, end in ranges:
        result.update(range(start, end + 1))
    return result


# ---------------------------------------------------------------------------
# 프레임 분할 기준 (확정안)
# ---------------------------------------------------------------------------
SPLITS = {
    "train": frames((1, 12), (14, 23), (32, 42), (52, 76)),
    "valid": frames((25, 30), (44, 50)),
    # frame 101은 원본에 존재하지 않으므로 79–100, 102–106 으로 정의한다.
    "test": frames((79, 100), (102, 106)),
}

# 제외 프레임과 제외 이유
EXCLUDED = {
    13: "seal이 손에 거의 가려져 bbox 기준이 애매함 (라벨 미부여, negative로도 사용하지 않음)",
    24: "validation 25–30 앞쪽 경계 buffer (유사 프레임 leakage 방지)",
    31: "validation 25–30 뒤쪽 경계 buffer (유사 프레임 leakage 방지)",
    43: "validation 44–50 앞쪽 경계 buffer (유사 프레임 leakage 방지)",
    51: "validation 44–50 뒤쪽 경계 buffer (50과 거의 같은 자세)",
    77: "카메라 구도 전환 중인 프레임 (train과 test 사이 buffer)",
    78: "이미 test 장면과 같은 원거리 구도 (train에 넣으면 test 누출)",
}

# Roboflow 파일명 예: train/images/261007_164405_000007_png.rf.<해시>.jpg
# 가운데 6자리 숫자(000007)가 프레임 번호다.
NAME_PATTERN = re.compile(r"^\d{6}_\d{6}_(\d{6})_png\.rf\.[0-9a-f]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def fail(message):
    """오류 메시지를 출력하고 종료한다. 아무 파일도 만들기 전에 검증이 끝나도록 사용한다."""
    print(f"\n[오류] {message}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1단계: 분할 기준 자체 검증 (프레임이 두 곳 이상에 속하지 않는지)
# ---------------------------------------------------------------------------
def check_split_definition():
    groups = dict(SPLITS)
    groups["excluded"] = set(EXCLUDED)
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = groups[a] & groups[b]
            if overlap:
                fail(f"{a}와 {b}에 동시에 속한 프레임이 있습니다: {sorted(overlap)}")
    return groups


# ---------------------------------------------------------------------------
# 2단계: ZIP 안의 이미지/label 목록을 읽고 1:1 대응을 확인
# ---------------------------------------------------------------------------
def read_zip_index(zf):
    """프레임 번호 -> {"image": zip 내부 경로, "label": zip 내부 경로} 사전을 만든다."""
    images, labels = {}, {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue  # 폴더 항목은 건너뛴다
        path = Path(name)
        folder = path.parent.name  # "images" 또는 "labels"
        if folder not in ("images", "labels"):
            continue  # data.yaml, README 등은 사용하지 않는다

        match = NAME_PATTERN.match(path.stem)
        if not match:
            fail(f"파일명 패턴이 예상과 다릅니다: {name}")
        frame = int(match.group(1))

        if folder == "images":
            if path.suffix.lower() not in IMAGE_EXTS:
                fail(f"이미지 확장자가 아닙니다: {name}")
            target = images
        else:
            if path.suffix != ".txt":
                fail(f"label 확장자가 .txt가 아닙니다: {name}")
            target = labels

        if frame in target:
            fail(f"같은 프레임 번호({frame})의 {folder} 파일이 두 개 있습니다: {target[frame]}, {name}")
        target[frame] = name

    # 이미지와 label이 정확히 짝을 이루는지 확인
    only_images = sorted(set(images) - set(labels))
    only_labels = sorted(set(labels) - set(images))
    if only_images:
        fail(f"label이 없는 이미지 프레임: {only_images}")
    if only_labels:
        fail(f"이미지가 없는 label 프레임: {only_labels}")
    for frame in images:
        if Path(images[frame]).stem != Path(labels[frame]).stem:
            fail(f"프레임 {frame}의 이미지와 label 파일명이 다릅니다: {images[frame]} / {labels[frame]}")

    return {f: {"image": images[f], "label": labels[f]} for f in sorted(images)}


# ---------------------------------------------------------------------------
# 3단계: YOLO annotation 형식 검증
#   한 줄 = "class_id x_center y_center width height" (좌표는 0~1로 정규화)
# ---------------------------------------------------------------------------
def parse_label(text, label_name):
    """label 내용을 검사하고 bbox 개수를 돌려준다. 빈 파일이면 0 (negative image)."""
    eps = 1e-6  # 부동소수점 반올림 오차 허용치
    count = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{label_name} {line_no}번째 줄 '{line.strip()}'"
        parts = line.split()
        if len(parts) != 5:
            fail(f"값이 5개가 아닙니다: {where}")
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            fail(f"숫자로 읽을 수 없습니다: {where}")
        if not 0 <= class_id < len(CLASS_NAMES):
            fail(f"클래스 번호가 범위를 벗어났습니다 (허용: 0~{len(CLASS_NAMES) - 1}): {where}")
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            fail(f"좌표에 NaN/inf가 있습니다: {where}")
        if not (0 < w <= 1 and 0 < h <= 1):
            fail(f"width/height는 0보다 크고 1 이하여야 합니다: {where}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            fail(f"중심 좌표가 0~1 범위를 벗어났습니다: {where}")
        if cx - w / 2 < -eps or cx + w / 2 > 1 + eps or cy - h / 2 < -eps or cy + h / 2 > 1 + eps:
            fail(f"bbox가 이미지 밖으로 나갑니다: {where}")
        count += 1
    return count


# ---------------------------------------------------------------------------
# 4단계: 출력 폴더 안전 처리
# ---------------------------------------------------------------------------
def prepare_output(overwrite):
    """기존 출력이 있으면 기본적으로 중단한다. --overwrite여도 이 스크립트가 만든 폴더만 지운다."""
    if OUTPUT_DIR.exists():
        if not overwrite:
            fail(
                f"출력 폴더가 이미 있습니다: {OUTPUT_DIR}\n"
                "       덮어쓰려면 --overwrite 옵션을 붙여 다시 실행하세요."
            )
        if not (OUTPUT_DIR / MARKER_FILE).exists():
            fail(
                f"{OUTPUT_DIR} 에 {MARKER_FILE} 이 없습니다.\n"
                "       이 스크립트가 만든 폴더가 아닐 수 있어 자동으로 지우지 않습니다. 직접 확인해주세요."
            )
    if DATA_YAML_PATH.exists() and not overwrite:
        fail(
            f"{DATA_YAML_PATH} 이 이미 있습니다.\n"
            "       덮어쓰려면 --overwrite 옵션을 붙여 다시 실행하세요."
        )


def file_sha256(path):
    """파일 전체의 SHA-256. 원본 ZIP 내용이 같은지 환경과 관계없이 확인하는 데 쓴다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_data_yaml():
    # Ultralytics는 path가 상대경로면 프로젝트가 아닌 자체 datasets 폴더 기준으로 찾기 때문에
    # 절대경로를 기록한다. 다른 PC에서는 이 스크립트를 다시 실행하면 경로가 맞게 갱신된다.
    lines = [
        "# scripts/split_dataset.py 가 자동 생성한 파일입니다. 직접 수정하지 마세요.",
        f"path: {OUTPUT_DIR.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    DATA_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="container_seal 데이터셋 프레임 기준 분할")
    parser.add_argument("--overwrite", action="store_true", help="이 스크립트가 만든 기존 출력을 교체")
    args = parser.parse_args()

    if not ZIP_PATH.exists():
        fail(f"입력 ZIP을 찾을 수 없습니다: {ZIP_PATH}")

    groups = check_split_definition()
    prepare_output(args.overwrite)

    # ZIP은 읽기 전용("r")으로만 연다. 원본은 절대 수정하지 않는다.
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        index = read_zip_index(zf)

        # 모든 ZIP 프레임이 정확히 하나의 그룹에만 속하는지 확인
        zip_frames = set(index)
        defined = set().union(*groups.values())
        not_assigned = sorted(zip_frames - defined)
        not_in_zip = sorted(defined - zip_frames)
        if not_assigned:
            fail(f"어느 split에도 정의되지 않은 프레임: {not_assigned}")
        if not_in_zip:
            fail(f"분할 기준에는 있지만 ZIP에 없는 프레임: {not_in_zip}")

        # 파일을 쓰기 전에 모든 label을 먼저 검증하고 bbox 개수를 센다
        box_counts = {}
        for frame, files in index.items():
            text = zf.read(files["label"]).decode("utf-8")
            box_counts[frame] = parse_label(text, files["label"])

        # 임시 폴더에 먼저 만들고, 전부 성공하면 최종 이름으로 바꾼다.
        # 중간에 실패해도 반쯤 만들어진 출력이 남지 않는다.
        tmp_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)  # 이전 실행이 중간에 끊겨 남은 임시 폴더
        for split in SPLITS:
            (tmp_dir / split / "images").mkdir(parents=True)
            (tmp_dir / split / "labels").mkdir(parents=True)

        # 이미지와 label은 항상 같은 프레임 번호로 함께 복사한다
        for split, split_frames in SPLITS.items():
            for frame in sorted(split_frames):
                for kind in ("image", "label"):
                    src = index[frame][kind]
                    dst = tmp_dir / split / f"{kind}s" / Path(src).name
                    dst.write_bytes(zf.read(src))

    # 복사 결과 재확인: split마다 images와 labels 파일명이 1:1로 맞는지
    for split, split_frames in SPLITS.items():
        img_stems = {p.stem for p in (tmp_dir / split / "images").iterdir()}
        lbl_stems = {p.stem for p in (tmp_dir / split / "labels").iterdir()}
        if img_stems != lbl_stems or len(img_stems) != len(split_frames):
            shutil.rmtree(tmp_dir)
            fail(f"{split} 복사 결과에서 이미지/label 대응이 맞지 않습니다.")

    # 기록용 표시 파일 (다음 --overwrite 때 이 스크립트가 만든 폴더인지 확인)
    zip_sha256 = file_sha256(ZIP_PATH)
    info = {
        "source_zip": ZIP_PATH.name,
        "source_zip_sha256": zip_sha256,
        "splits": {s: sorted(f) for s, f in SPLITS.items()},
        "excluded": {str(f): r for f, r in sorted(EXCLUDED.items())},
    }
    # write_text는 Windows에서 줄바꿈을 CRLF로 바꾸므로, OS와 관계없이 LF로 저장되도록 바이트로 쓴다
    text = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    (tmp_dir / MARKER_FILE).write_bytes(text.encode("utf-8"))

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)  # prepare_output에서 이 스크립트가 만든 폴더임을 확인함
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(OUTPUT_DIR)
    write_data_yaml()

    # -----------------------------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------------------------
    print(f"입력 ZIP : {ZIP_PATH}")
    print(f"ZIP SHA-256: {zip_sha256}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"data.yaml: {DATA_YAML_PATH}\n")

    print(f"{'split':<10}{'images':>8}{'boxes':>8}{'empty':>8}")
    total_images = total_boxes = 0
    for split, split_frames in SPLITS.items():
        n_img = len(split_frames)
        n_box = sum(box_counts[f] for f in split_frames)
        n_empty = sum(1 for f in split_frames if box_counts[f] == 0)
        total_images += n_img
        total_boxes += n_box
        print(f"{split:<10}{n_img:>8}{n_box:>8}{n_empty:>8}")
    print(f"{'(합계)':<9}{total_images:>8}{total_boxes:>8}")

    print(f"\n제외된 프레임 ({len(EXCLUDED)}장, 출력에 포함하지 않음):")
    for frame, reason in sorted(EXCLUDED.items()):
        print(f"  frame {frame:>3} (bbox {box_counts[frame]}개): {reason}")

    print(
        f"\n검증 완료: ZIP 이미지 {len(index)}장 = 사용 {total_images}장 + 제외 {len(EXCLUDED)}장, "
        "모든 프레임이 정확히 한 그룹에만 속함, 이미지/label 1:1 대응, YOLO 형식 오류 없음"
    )


if __name__ == "__main__":
    main()
