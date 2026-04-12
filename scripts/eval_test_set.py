from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detection and segmentation models on the test split and save a unified metrics summary."
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate, e.g. test/val.")
    parser.add_argument("--det-model", type=str, default="", help="Detection model path (.pt).")
    parser.add_argument("--det-data", type=Path, default=Path("GZ-DET.yaml"), help="Detection data YAML.")
    parser.add_argument("--seg-model", type=str, default="", help="Segmentation model path (.pt).")
    parser.add_argument("--seg-data", type=Path, default=Path("GZ-DET-seg.yaml"), help="Segmentation data YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--device", type=str, default="", help="Device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for val.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS in val.")
    parser.add_argument("--project", type=Path, default=Path("runs/gyu_eval"), help="Output root.")
    parser.add_argument("--name", type=str, default="test_eval", help="Run name.")
    parser.add_argument("--plots", action="store_true", help="Save PR/confusion plots.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO/LVIS json output when applicable.")
    return parser.parse_args()


def ensure_split_exists(data_yaml: Path, split: str) -> None:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if split not in cfg or not cfg.get(split):
        raise ValueError(f"Split '{split}' is missing in {data_yaml}. Please add '{split}: <path>'.")


def normalize_scalars(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for k, v in data.items():
        if hasattr(v, "item"):
            normalized[k] = float(v.item())
        elif isinstance(v, (int, float, str, bool)) or v is None:
            normalized[k] = v
        else:
            normalized[k] = str(v)
    return normalized


def run_detection(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Failed to import ultralytics for detection eval. Ensure OpenCV is installed.") from e
    ensure_split_exists(args.det_data, args.split)
    model = YOLO(args.det_model)
    metrics = model.val(
        data=str(args.det_data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=f"{args.name}_det",
        exist_ok=True,
        plots=args.plots,
        save_json=args.save_json,
        verbose=True,
    )
    return {
        "model": args.det_model,
        "data_yaml": str(args.det_data),
        "save_dir": str(metrics.save_dir),
        "metrics": normalize_scalars(metrics.results_dict),
        "speed_ms": normalize_scalars(metrics.speed),
    }


def run_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Failed to import ultralytics for segmentation eval. Ensure OpenCV is installed.") from e
    ensure_split_exists(args.seg_data, args.split)
    model = YOLO(args.seg_model)
    metrics = model.val(
        data=str(args.seg_data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=f"{args.name}_seg",
        exist_ok=True,
        plots=args.plots,
        save_json=args.save_json,
        verbose=True,
    )
    return {
        "model": args.seg_model,
        "data_yaml": str(args.seg_data),
        "save_dir": str(metrics.save_dir),
        "metrics": normalize_scalars(metrics.results_dict),
        "speed_ms": normalize_scalars(metrics.speed),
    }


def main() -> None:
    args = parse_args()
    if not args.det_model and not args.seg_model:
        raise ValueError("At least one model must be provided: --det-model and/or --seg-model.")

    output_root = args.project / args.name
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
    }

    if args.det_model:
        summary["detection"] = run_detection(args)
    if args.seg_model:
        summary["segmentation"] = run_segmentation(args)

    summary_path = output_root / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
