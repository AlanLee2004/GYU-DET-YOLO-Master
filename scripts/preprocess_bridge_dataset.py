from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python") from e
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess bridge-disease dataset with illumination correction and light deblurring."
    )
    parser.add_argument("--data-yaml", type=Path, default=Path("GZ-DET.yaml"), help="Input YOLO data YAML file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/GYU-DET-enhanced"),
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path("GZ-DET-enhanced.yaml"),
        help="Output YOLO data YAML file for the enhanced dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output root if it already exists.")
    parser.add_argument("--clahe-clip", type=float, default=2.5, help="CLAHE clip limit.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile size.")
    parser.add_argument("--gamma-min", type=float, default=0.80, help="Lower bound of adaptive gamma.")
    parser.add_argument("--gamma-max", type=float, default=1.30, help="Upper bound of adaptive gamma.")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=90.0,
        help="Variance-of-Laplacian threshold under which unsharp masking is applied.",
    )
    parser.add_argument("--unsharp-amount", type=float, default=0.35, help="Unsharp masking amount.")
    return parser.parse_args()


def _as_unix_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _resolve_split_path(data_cfg: dict, split_key: str, yaml_file: Path) -> Path | None:
    split_value = data_cfg.get(split_key)
    if split_value is None:
        return None
    if isinstance(split_value, (list, tuple)):
        raise ValueError(f"Unsupported split format for '{split_key}': list/tuple is not supported by this script.")

    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path

    if data_cfg.get("path"):
        return (Path(data_cfg["path"]) / split_path).resolve()
    return (yaml_file.parent / split_path).resolve()


def _infer_split_layout(split_path: Path) -> tuple[Path, Path | None, str]:
    # Case 1: split_path points to split root: train/{images,labels}
    if (split_path / "images").is_dir():
        images_dir = split_path / "images"
        labels_dir = split_path / "labels" if (split_path / "labels").is_dir() else None
        yaml_rel = split_path.name
        return images_dir, labels_dir, yaml_rel

    # Case 2: split_path points to split images dir: train/images
    if split_path.name == "images":
        images_dir = split_path
        labels_dir = split_path.parent / "labels" if (split_path.parent / "labels").is_dir() else None
        yaml_rel = f"{split_path.parent.name}/images"
        return images_dir, labels_dir, yaml_rel

    # Case 3: split_path points directly to image directory with no explicit labels structure
    return split_path, None, split_path.name


def _adaptive_gamma_lut(gray: np.ndarray, gamma_min: float, gamma_max: float) -> tuple[np.ndarray, float]:
    mean_luma = max(float(gray.mean()) / 255.0, 1e-3)
    target_luma = 0.5
    gamma = float(np.clip(np.log(target_luma) / np.log(mean_luma), gamma_min, gamma_max))
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return lut, gamma


def enhance_image(
    image: np.ndarray,
    clahe_clip: float,
    clahe_grid: int,
    gamma_min: float,
    gamma_max: float,
    blur_threshold: float,
    unsharp_amount: float,
) -> tuple[np.ndarray, float, float]:
    cv2 = _require_cv2()

    # Illumination normalization: CLAHE on L channel.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Adaptive gamma based on current luminance.
    gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    lut, gamma = _adaptive_gamma_lut(gray, gamma_min, gamma_max)
    corrected = cv2.LUT(clahe_img, lut)

    # Light deblurring: only sharpen images detected as blurry.
    blur_score = float(cv2.Laplacian(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if blur_score < blur_threshold:
        smooth = cv2.GaussianBlur(corrected, (0, 0), sigmaX=1.2)
        corrected = cv2.addWeighted(corrected, 1.0 + unsharp_amount, smooth, -unsharp_amount, 0)

    return corrected, blur_score, gamma


def process_split(
    split_name: str,
    split_path: Path,
    output_root: Path,
    clahe_clip: float,
    clahe_grid: int,
    gamma_min: float,
    gamma_max: float,
    blur_threshold: float,
    unsharp_amount: float,
) -> tuple[str, int]:
    cv2 = _require_cv2()

    images_dir, labels_dir, yaml_rel = _infer_split_layout(split_path)
    if not images_dir.exists():
        raise FileNotFoundError(f"[{split_name}] image directory not found: {images_dir}")

    dst_images_dir = output_root / yaml_rel / "images" if images_dir.name != "images" else output_root / yaml_rel
    if images_dir.name == "images" and yaml_rel.endswith("/images"):
        dst_images_dir = output_root / yaml_rel
    elif images_dir.name == "images":
        dst_images_dir = output_root / yaml_rel / "images"
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    if labels_dir is not None:
        dst_labels_dir = (
            output_root / Path(yaml_rel).parent / "labels"
            if Path(yaml_rel).name == "images"
            else output_root / yaml_rel / "labels"
        )
        shutil.copytree(labels_dir, dst_labels_dir, dirs_exist_ok=True)

    processed = 0
    skipped = 0
    for src_img in sorted(images_dir.rglob("*")):
        if src_img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = src_img.relative_to(images_dir)
        dst_img = dst_images_dir / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src_img))
        if img is None:
            skipped += 1
            continue
        out, _, _ = enhance_image(
            img,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            blur_threshold=blur_threshold,
            unsharp_amount=unsharp_amount,
        )
        cv2.imwrite(str(dst_img), out)
        processed += 1

    print(f"[{split_name}] processed={processed}, skipped={skipped}, output={dst_images_dir}")
    return yaml_rel, processed


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    new_cfg = dict(data_cfg)
    total_processed = 0
    split_map = {}
    for split_key in ("train", "val", "test"):
        split_path = _resolve_split_path(data_cfg, split_key, args.data_yaml)
        if split_path is None:
            continue
        yaml_rel, split_count = process_split(
            split_name=split_key,
            split_path=split_path,
            output_root=args.output_root,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            blur_threshold=args.blur_threshold,
            unsharp_amount=args.unsharp_amount,
        )
        split_map[split_key] = _as_unix_path((args.output_root / yaml_rel).resolve())
        total_processed += split_count

    for split_key, split_value in split_map.items():
        new_cfg[split_key] = split_value
    new_cfg.pop("path", None)  # use absolute paths for generated config

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[DONE] total_processed={total_processed}")
    print(f"[DONE] output_yaml={args.output_yaml.resolve()}")


if __name__ == "__main__":
    main()
