"""split_dataset.py 검증 로직 테스트.

실제 이미지·라벨 없이, Roboflow export와 같은 파일명 규칙을 가진 작은 가짜 ZIP으로
분할 정의, 라벨 형식 검사, 출력 폴더 보호, data.yaml 생성을 확인한다.
Python 표준 라이브러리만 사용하므로 ultralytics나 torch가 없어도 실행된다.

실행 (프로젝트 루트에서):
    python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import common  # noqa: E402
import plot_split  # noqa: E402
import split_dataset as sd  # noqa: E402

ALL_FRAMES = sorted(set().union(*sd.SPLITS.values()) | set(sd.EXCLUDED))
BOX_LINE = "0 0.5 0.5 0.1 0.2\n"


def image_name(frame, folder="train"):
    return f"{folder}/images/261007_164405_{frame:06d}_png.rf.{frame:08x}.jpg"


def label_name(frame, folder="train"):
    return f"{folder}/labels/261007_164405_{frame:06d}_png.rf.{frame:08x}.txt"


def make_zip(path, frames=ALL_FRAMES, labels=None, extra=()):
    """frames마다 이미지·라벨 쌍을 넣은 ZIP을 만든다. labels로 프레임별 라벨 내용을 바꿀 수 있다."""
    labels = labels or {}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.yaml", "names: [container_seal]\n")  # 분할에 쓰지 않는 파일
        for frame in frames:
            zf.writestr(image_name(frame), b"fake-jpeg")
            zf.writestr(label_name(frame), labels.get(frame, BOX_LINE))
        for name, content in extra:
            zf.writestr(name, content)


def quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


class SplitDefinitionTests(unittest.TestCase):
    def test_every_frame_belongs_to_exactly_one_group(self):
        groups = sd.check_split_definition()
        self.assertEqual(sum(len(frames) for frames in groups.values()), len(ALL_FRAMES))
        self.assertEqual(ALL_FRAMES, [f for f in range(1, 107) if f != 101])

    def test_split_sizes_match_readme(self):
        sizes = {name: len(frames) for name, frames in sd.SPLITS.items()}
        self.assertEqual(sizes, {"train": 58, "valid": 13, "test": 27})
        self.assertEqual(len(sd.EXCLUDED), 7)

    def test_test_frames_come_after_all_train_and_valid_frames(self):
        # 원거리 test 장면은 근접 장면(train/valid)과 시간상 완전히 분리되어 있어야 한다.
        self.assertGreater(min(sd.SPLITS["test"]), max(sd.SPLITS["train"] | sd.SPLITS["valid"]) + 1)

    def test_validation_blocks_are_buffered_from_train(self):
        # validation 프레임의 바로 앞뒤 프레임은 train에 없어야 한다 (경계 buffer).
        for frame in sd.SPLITS["valid"]:
            for neighbor in (frame - 1, frame + 1):
                self.assertNotIn(neighbor, sd.SPLITS["train"], f"frame {frame}의 이웃 {neighbor}")

    def test_overlapping_definition_is_rejected(self):
        bad = dict(sd.SPLITS, valid=sd.SPLITS["valid"] | {12})  # 12는 train
        with mock.patch.object(sd, "SPLITS", bad), self.assertRaises(SystemExit):
            quiet(sd.check_split_definition)

    def test_split_hash_matches_recorded_runs(self):
        # README와 experiments.csv에 기록된 split_hash. 분할을 바꾸면 이 값도 바뀌어야 한다.
        info = {
            "splits": {s: sorted(f) for s, f in sd.SPLITS.items()},
            "excluded": {str(f): r for f, r in sd.EXCLUDED.items()},
        }
        self.assertEqual(common.split_hash(info), "6f8a98888a3b")

    def test_split_hash_ignores_order_and_reasons(self):
        info = {
            "splits": {s: sorted(f) for s, f in sd.SPLITS.items()},
            "excluded": {str(f): r for f, r in sd.EXCLUDED.items()},
        }
        shuffled = {
            "splits": {s: sorted(f, reverse=True) for s, f in reversed(list(sd.SPLITS.items()))},
            "excluded": {str(f): "다른 설명" for f in sd.EXCLUDED},
        }
        self.assertEqual(common.split_hash(info), common.split_hash(shuffled))


class ParseLabelTests(unittest.TestCase):
    def test_counts_boxes_and_ignores_blank_lines(self):
        self.assertEqual(sd.parse_label("0 0.5 0.5 0.1 0.2\n\n0 0.2 0.3 0.1 0.1\n", "x.txt"), 2)

    def test_empty_label_is_negative_image(self):
        self.assertEqual(sd.parse_label("", "x.txt"), 0)

    def test_box_touching_image_edge_is_allowed(self):
        self.assertEqual(sd.parse_label("0 0.05 0.95 0.1 0.1\n", "x.txt"), 1)

    def test_invalid_lines_are_rejected(self):
        cases = {
            "값 4개": "0 0.5 0.5 0.1",
            "숫자 아님": "0 a 0.5 0.1 0.1",
            "클래스 범위 밖": "1 0.5 0.5 0.1 0.1",
            "음수 클래스": "-1 0.5 0.5 0.1 0.1",
            "폭 0": "0 0.5 0.5 0 0.1",
            "폭 1 초과": "0 0.5 0.5 1.2 0.1",
            "NaN": "0 nan 0.5 0.1 0.1",
            "중심 범위 밖": "0 1.2 0.5 0.1 0.1",
            "이미지 밖으로 나감": "0 0.98 0.5 0.1 0.1",
        }
        for name, line in cases.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                quiet(sd.parse_label, line + "\n", "x.txt")


class ZipIndexTests(unittest.TestCase):
    def index(self, entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name in entries:
                zf.writestr(name, "")
        with zipfile.ZipFile(buffer) as zf:
            return quiet(sd.read_zip_index, zf)

    def test_pairs_images_and_labels_by_frame(self):
        index = self.index([image_name(1), label_name(1), image_name(2, "valid"), label_name(2, "valid")])
        self.assertEqual(sorted(index), [1, 2])
        self.assertEqual(index[2]["label"], label_name(2, "valid"))

    def test_image_without_label_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.index([image_name(1), image_name(2), label_name(2)])

    def test_label_without_image_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.index([image_name(1), label_name(1), label_name(2)])

    def test_same_frame_twice_is_rejected(self):
        # Roboflow가 같은 프레임을 train과 valid에 각각 넣은 경우
        with self.assertRaises(SystemExit):
            self.index([image_name(1), label_name(1), image_name(1, "valid"), label_name(1, "valid")])

    def test_unexpected_file_name_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.index(["train/images/frame_0001.jpg", label_name(1)])


class SplitPipelineTests(unittest.TestCase):
    """가짜 ZIP으로 main()을 끝까지 실행한다. 경로 상수는 임시 폴더로 바꾼다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.zip_path = self.root / "raw" / "export.zip"
        self.zip_path.parent.mkdir()
        self.output = self.root / "processed" / "container_seal_v2"
        self.data_yaml = self.root / "configs" / "data.yaml"
        for name, value in (("ZIP_PATH", self.zip_path), ("OUTPUT_DIR", self.output),
                            ("DATA_YAML_PATH", self.data_yaml)):
            patcher = mock.patch.object(sd, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *args):
        with mock.patch.object(sys, "argv", ["split_dataset.py", *args]):
            quiet(sd.main)

    def assert_fails_without_output(self, *args):
        with self.assertRaises(SystemExit):
            self.run_main(*args)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.data_yaml.exists())

    def test_creates_splits_marker_and_data_yaml(self):
        make_zip(self.zip_path, labels={13: "", 48: ""})
        self.run_main()

        for split, frames in sd.SPLITS.items():
            images = sorted((self.output / split / "images").iterdir())
            labels = sorted((self.output / split / "labels").iterdir())
            self.assertEqual(len(images), len(frames), split)
            self.assertEqual([p.stem for p in images], [p.stem for p in labels], split)

        copied = {int(p.stem.split("_")[2]) for p in self.output.rglob("*.jpg")}
        self.assertEqual(copied, set().union(*sd.SPLITS.values()))
        self.assertFalse(copied & set(sd.EXCLUDED), "제외 프레임이 출력에 들어감")

        marker = json.loads((self.output / sd.MARKER_FILE).read_text(encoding="utf-8"))
        self.assertEqual(marker["splits"], {s: sorted(f) for s, f in sd.SPLITS.items()})
        self.assertEqual(len(marker["source_zip_sha256"]), 64)
        self.assertNotIn(b"\r\n", (self.output / sd.MARKER_FILE).read_bytes())

        yaml_text = self.data_yaml.read_text(encoding="utf-8")
        self.assertIn(f"path: {self.output.as_posix()}", yaml_text)
        self.assertIn("val: valid/images", yaml_text)
        self.assertFalse(self.output.with_name(self.output.name + ".tmp").exists())

    def test_existing_output_is_not_overwritten_without_flag(self):
        make_zip(self.zip_path)
        self.run_main()
        sentinel = self.output / "train" / "labels" / "keep.txt"
        sentinel.write_text("x")
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertTrue(sentinel.exists())

    def test_overwrite_replaces_previous_output(self):
        make_zip(self.zip_path)
        self.run_main()
        sentinel = self.output / "train" / "labels" / "keep.txt"
        sentinel.write_text("x")
        self.run_main("--overwrite")
        self.assertFalse(sentinel.exists())
        self.assertTrue((self.output / sd.MARKER_FILE).exists())

    def test_overwrite_refuses_folder_not_made_by_script(self):
        make_zip(self.zip_path)
        self.output.mkdir(parents=True)
        (self.output / "important.txt").write_text("x")
        with self.assertRaises(SystemExit):
            self.run_main("--overwrite")
        self.assertTrue((self.output / "important.txt").exists())

    def test_missing_zip_is_rejected(self):
        self.assert_fails_without_output()

    def test_frame_missing_from_zip_is_rejected(self):
        make_zip(self.zip_path, frames=[f for f in ALL_FRAMES if f != 50])
        self.assert_fails_without_output()

    def test_frame_not_in_any_split_is_rejected(self):
        make_zip(self.zip_path, extra=[(image_name(107), b"x"), (label_name(107), BOX_LINE)])
        self.assert_fails_without_output()

    def test_invalid_label_stops_before_writing_files(self):
        make_zip(self.zip_path, labels={30: "0 0.5 0.5 0.1\n"})
        self.assert_fails_without_output()


class SplitDiagramTests(unittest.TestCase):
    def test_committed_diagram_matches_split_definition(self):
        committed = plot_split.OUTPUT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(committed, plot_split.render(),
                         "assets/split_timeline.svg가 분할 정의와 다릅니다. python scripts/plot_split.py 로 다시 만드세요.")


if __name__ == "__main__":
    unittest.main()
