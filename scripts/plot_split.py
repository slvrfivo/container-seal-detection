"""
split_dataset.py의 분할 정의(SPLITS, EXCLUDED)로 프레임 번호별 분할 그림(SVG)을 만든다.

이미지나 라벨 없이 코드에 적힌 프레임 번호만 쓴다. 분할 정의를 바꾸면 이 스크립트를
다시 실행해서 그림을 갱신해야 하고, tests/test_split_dataset.py가 그림과 코드가 같은지 확인한다.

사용법 (프로젝트 루트에서):
    python scripts/plot_split.py            # assets/split_timeline.svg 갱신
    python scripts/plot_split.py --check    # 파일이 코드와 같은지만 확인 (다르면 종료 코드 1)
"""

import argparse
import sys

from split_dataset import EXCLUDED, PROJECT_ROOT, SPLITS

OUTPUT_PATH = PROJECT_ROOT / "assets" / "split_timeline.svg"

FIRST_FRAME = 1
LAST_FRAME = max(max(frames) for frames in SPLITS.values())

CELL = 8          # 프레임 1개의 가로 폭 [px]
LEFT = 24
STRIP_Y = 46
STRIP_H = 30
WIDTH = LEFT * 2 + (LAST_FRAME - FIRST_FRAME + 1) * CELL
HEIGHT = 150
FONT = "'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif"

# 흰 글씨가 잘 보이는 진한 색. 배경은 흰색으로 고정해서 다크 모드에서도 같은 그림으로 보이게 한다.
COLORS = {
    "train": "#3B6FB6",
    "valid": "#C2570C",
    "test": "#2E7D4F",
    "excluded": "#8C8C8C",
}
SHORT_NAMES = {"train": "train", "valid": "val", "test": "test"}
LEGEND_NAMES = {"train": "train", "valid": "validation", "test": "test", "excluded": "제외"}


def frame_groups():
    """프레임 번호 -> 그룹 이름 (train / valid / test / excluded). 원본에 없는 프레임은 빠진다."""
    groups = {}
    for name, frames in SPLITS.items():
        for frame in frames:
            groups[frame] = name
    for frame in EXCLUDED:
        groups[frame] = "excluded"
    return groups


def runs(groups):
    """같은 그룹이 연속된 구간 목록 [(그룹, 시작 프레임, 끝 프레임)]."""
    result = []
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        group = groups.get(frame)
        if result and result[-1][0] == group and result[-1][2] == frame - 1:
            result[-1] = (group, result[-1][1], frame)
        else:
            result.append((group, frame, frame))
    return [run for run in result if run[0] is not None]


def text_width(content, size=12):
    """범례 간격용 대략적인 글자 폭 [px]. 한글은 영문·숫자보다 넓게 잡는다."""
    return sum(size if ord(ch) > 0x2E80 else size * 0.6 for ch in content)


def x_of(frame):
    return LEFT + (frame - FIRST_FRAME) * CELL


def text(x, y, content, size=12, anchor="middle", color="#333333", weight="normal"):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}">{content}</text>')


def bracket(start, end, label):
    """장면 구간 위에 꺾쇠와 이름을 그린다."""
    x1, x2 = x_of(start), x_of(end) + CELL - 1
    y = STRIP_Y - 8
    return [
        f'<path d="M{x1:g} {y + 4:g} V{y:g} H{x2:g} V{y + 4:g}" fill="none" stroke="#666666" stroke-width="1"/>',
        text((x1 + x2) / 2, y - 6, label, size=12),
    ]


def render():
    groups = frame_groups()
    missing = [f for f in range(FIRST_FRAME, LAST_FRAME + 1) if f not in groups]

    # 장면 구간: 근접 장면 = train·validation이 있는 구간, 원거리 장면 = test 구간
    near_end = max(SPLITS["train"] | SPLITS["valid"])
    far_start, far_end = min(SPLITS["test"]), max(SPLITS["test"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="6" fill="#FFFFFF"/>',
    ]
    parts += bracket(FIRST_FRAME, near_end, f"근접 장면 (frame {FIRST_FRAME}–{near_end})")
    parts += bracket(far_start, far_end, f"원거리 장면 (frame {far_start}–{far_end})")

    # 프레임 칸
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        x = x_of(frame)
        if frame in groups:
            parts.append(f'<rect x="{x}" y="{STRIP_Y}" width="{CELL - 1}" height="{STRIP_H}" '
                         f'fill="{COLORS[groups[frame]]}"/>')
        else:  # 원본에 없는 프레임 (101)
            parts.append(f'<rect x="{x + 0.5:g}" y="{STRIP_Y + 0.5:g}" width="{CELL - 2}" height="{STRIP_H - 1}" '
                         f'fill="none" stroke="#8C8C8C" stroke-dasharray="2 2"/>')

    # 충분히 긴 구간에는 이름을 적는다
    for group, start, end in runs(groups):
        width = (end - start + 1) * CELL
        if group in SHORT_NAMES and width >= 40:
            parts.append(text(x_of(start) + width / 2, STRIP_Y + STRIP_H / 2 + 4, SHORT_NAMES[group],
                              size=11, color="#FFFFFF", weight="bold"))

    # 프레임 번호 눈금
    ticks = [FIRST_FRAME] + list(range(10, LAST_FRAME + 1, 10)) + [LAST_FRAME]
    axis_y = STRIP_Y + STRIP_H
    for frame in ticks:
        cx = x_of(frame) + (CELL - 1) / 2
        parts.append(f'<line x1="{cx:g}" y1="{axis_y + 1}" x2="{cx:g}" y2="{axis_y + 5}" stroke="#666666"/>')
        parts.append(text(cx, axis_y + 17, frame, size=10, color="#666666"))

    # 범례 (장 수는 분할 정의에서 계산)
    counts = {name: len(frames) for name, frames in SPLITS.items()}
    counts["excluded"] = len(EXCLUDED)
    x, legend_y = LEFT, HEIGHT - 22
    for group in ("train", "valid", "test", "excluded"):
        label = f"{LEGEND_NAMES[group]} {counts[group]}장"
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{COLORS[group]}"/>')
        parts.append(text(x + 18, legend_y, label, anchor="start"))
        x += 18 + text_width(label) + 28
    if missing:
        label = f"원본 없음 (frame {', '.join(str(f) for f in missing)})"
        parts.append(f'<rect x="{x + 0.5}" y="{legend_y - 9.5}" width="11" height="11" fill="none" '
                     f'stroke="#8C8C8C" stroke-dasharray="2 2"/>')
        parts.append(text(x + 18, legend_y, label, anchor="start"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="프레임 분할 그림(SVG) 생성")
    parser.add_argument("--check", action="store_true", help="파일을 쓰지 않고 코드와 같은지만 확인")
    args = parser.parse_args()

    svg = render()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n") if OUTPUT_PATH.exists() else None
        if current != svg:
            sys.exit(f"[오류] {OUTPUT_PATH} 가 분할 정의와 다릅니다. python scripts/plot_split.py 로 다시 만드세요.")
        print(f"[확인] {OUTPUT_PATH} 는 분할 정의와 같습니다.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(svg.encode("utf-8"))  # OS와 관계없이 LF로 저장
    print(f"[완료] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
