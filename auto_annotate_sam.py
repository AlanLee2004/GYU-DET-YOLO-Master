import os
import shutil
from pathlib import Path

import cv2
from ultralytics import SAM
from ultralytics.data import YOLODataset
from ultralytics.utils import YAML
from ultralytics.utils.ops import xywh2xyxy

DATA_CFG = "GZ-DET.yaml"
SAM_MODEL = "sam_b.pt"  # or "mobile_sam.pt"
DEVICE = ""  # "", "cpu", "0"

# Supports both common validation keys in YAML.
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid"),
}
BACKUP_DET_LABELS = True
REPLACE_LABELS_WITH_SEG = True


def resolve_images_dir(split_path: str) -> Path:
    p = Path(split_path)
    if p.name == "images":
        return p
    images_subdir = p / "images"
    if images_subdir.exists():
        return images_subdir
    return p


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    sa = f"{os.sep}images{os.sep}"
    sb = f"{os.sep}labels{os.sep}"
    path_str = str(images_dir)
    if sa in path_str:
        return Path(path_str.replace(sa, sb))
    return images_dir.parent / "labels"


def backup_detection_labels(labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing_label_files = list(labels_dir.rglob("*.txt"))
    if BACKUP_DET_LABELS and existing_label_files:
        backup_dir = labels_dir.parent / f"{labels_dir.name}_bbox_backup"
        if not backup_dir.exists():
            shutil.copytree(labels_dir, backup_dir)
            print(f"[INFO] Backed up detection labels to: {backup_dir}")


def get_split_path(data_cfg: dict, split_name: str) -> tuple[str | None, str | None]:
    for key in SPLIT_ALIASES[split_name]:
        value = data_cfg.get(key)
        if value:
            return key, value
    return None, None


def image_to_output_label(images_dir: Path, output_dir: Path, im_file: str) -> Path:
    image_path = Path(im_file)
    try:
        rel = image_path.relative_to(images_dir)
        return (output_dir / rel).with_suffix(".txt")
    except ValueError:
        return output_dir / f"{image_path.stem}.txt"


def force_bbox_to_segment(images_dir: Path, output_dir: Path, sam_model_name: str, device: str | None = None) -> int:
    dataset = YOLODataset(images_dir, data=dict(names=list(range(1000)), channels=3))
    sam_model = SAM(sam_model_name)
    generated_files = 0

    for label in dataset.labels:
        txt_file = image_to_output_label(images_dir, output_dir, label["im_file"])
        txt_file.parent.mkdir(parents=True, exist_ok=True)

        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:
            txt_file.write_text("", encoding="utf-8")
            continue

        boxes = boxes.copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        image = cv2.imread(label["im_file"])
        if image is None:
            print(f"[WARN] Failed to read image: {label['im_file']}")
            txt_file.write_text("", encoding="utf-8")
            continue

        sam_results = sam_model(
            image,
            bboxes=xywh2xyxy(boxes),
            verbose=False,
            save=False,
            device=device,
        )
        segments = sam_results[0].masks.xyn if sam_results and sam_results[0].masks is not None else []
        classes = label["cls"].reshape(-1).astype(int)

        n = min(len(segments), len(classes))
        lines = []
        for i in range(n):
            s = segments[i]
            if len(s) == 0:
                continue
            line = (int(classes[i]), *s.reshape(-1).tolist())
            lines.append(("%g " * len(line)).rstrip() % line)

        txt_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        generated_files += 1

    return generated_files


def main() -> None:
    data_cfg = YAML.load(DATA_CFG)

    for split in ("train", "val"):
        split_key, split_path = get_split_path(data_cfg, split)
        if not split_path:
            print(f"[WARN] Split '{split}' is not defined in {DATA_CFG}, skip.")
            continue

        images_dir = resolve_images_dir(split_path)
        labels_dir = labels_dir_from_images_dir(images_dir)

        print(f"[INFO] Split: {split} (yaml key: {split_key})")
        print(f"[INFO] Images: {images_dir}")
        print(f"[INFO] Labels: {labels_dir}")

        backup_detection_labels(labels_dir)

        temp_seg_dir = labels_dir.parent / f"{labels_dir.name}_sam_tmp"
        if temp_seg_dir.exists():
            shutil.rmtree(temp_seg_dir)
        temp_seg_dir.mkdir(parents=True, exist_ok=True)

        generated_count = force_bbox_to_segment(
            images_dir=images_dir,
            output_dir=temp_seg_dir,
            sam_model_name=SAM_MODEL,
            device=DEVICE or None,
        )

        generated_seg_files = list(temp_seg_dir.rglob("*.txt"))
        if generated_count == 0 or not generated_seg_files:
            print(f"[WARN] No segmentation labels generated for split '{split}', skip replacing labels.")
            shutil.rmtree(temp_seg_dir, ignore_errors=True)
            continue

        if REPLACE_LABELS_WITH_SEG:
            for txt_file in labels_dir.rglob("*.txt"):
                txt_file.unlink()
            print(f"[INFO] Cleared old detection labels in: {labels_dir}")

            for seg_file in generated_seg_files:
                target_file = labels_dir / seg_file.relative_to(temp_seg_dir)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(seg_file), str(target_file))
            print(f"[INFO] Replaced labels with SAM segments: {labels_dir}")
            shutil.rmtree(temp_seg_dir, ignore_errors=True)
        else:
            print(f"[INFO] SAM segment labels kept in temporary output: {temp_seg_dir}")

    print("[DONE] SAM auto-annotation finished.")


if __name__ == "__main__":
    main()
