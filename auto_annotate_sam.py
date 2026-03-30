import os
import shutil
from pathlib import Path

from ultralytics.data.annotator import auto_annotate
from ultralytics.utils import YAML

DATA_CFG = "GZ-DET.yaml"
DET_MODEL = "runs/gyu_det/p2_imgsz960/weights/best.pt"
SAM_MODEL = "sam_b.pt"  # or "mobile_sam.pt"

DEVICE = ""  # "", "cpu", "0"
IMGSZ = 1280
CONF = 0.25
IOU = 0.45
MAX_DET = 300

SPLITS = ("train", "val")
BACKUP_DET_LABELS = True
CLEAR_OLD_LABELS = True


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


def backup_and_prepare_labels_dir(labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing_label_files = list(labels_dir.rglob("*.txt"))
    if BACKUP_DET_LABELS and existing_label_files:
        backup_dir = labels_dir.parent / f"{labels_dir.name}_bbox_backup"
        if not backup_dir.exists():
            shutil.copytree(labels_dir, backup_dir)
            print(f"[INFO] Backed up detection labels to: {backup_dir}")

    if CLEAR_OLD_LABELS:
        for txt_file in labels_dir.rglob("*.txt"):
            txt_file.unlink()
        print(f"[INFO] Cleared old labels in: {labels_dir}")


def main() -> None:
    data_cfg = YAML.load(DATA_CFG)

    for split in SPLITS:
        split_path = data_cfg.get(split)
        if not split_path:
            print(f"[WARN] Split '{split}' is not defined in {DATA_CFG}, skip.")
            continue

        images_dir = resolve_images_dir(split_path)
        labels_dir = labels_dir_from_images_dir(images_dir)

        print(f"[INFO] Split: {split}")
        print(f"[INFO] Images: {images_dir}")
        print(f"[INFO] Labels: {labels_dir}")

        backup_and_prepare_labels_dir(labels_dir)

        auto_annotate(
            data=images_dir,
            det_model=DET_MODEL,
            sam_model=SAM_MODEL,
            device=DEVICE,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            output_dir=labels_dir,
        )

    print("[DONE] SAM auto-annotation finished.")


if __name__ == "__main__":
    main()
