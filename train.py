from ultralytics import YOLO
from pathlib import Path

try:
    import albumentations as A
except ImportError:
    A = None

MODEL_CFG = "ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml"
PRETRAINED_WEIGHTS = "YOLO-Master-EsMoE-M.pt"
DATA_CFG = "GZ-DET-enhanced.yaml" if Path("GZ-DET-enhanced.yaml").exists() else "GZ-DET.yaml"
TRAIN_IMGSZ = 640  # low-VRAM fallback; try 768/960/1280 when memory allows
TRAIN_BATCH = -1  # AutoBatch: choose the largest safe batch for current GPU memory


def build_bridge_augmentations():
    """Bridge-disease specific augmentations for uneven illumination and motion blur robustness."""
    if A is None:
        print("[WARN] albumentations is not installed, skip custom bridge augmentations.")
        return None

    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 9), p=1.0),
                A.Blur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.25,
        ),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.20),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        A.RandomGamma(gamma_limit=(80, 125), p=0.20),
        A.OneOf(
            [
                A.GaussNoise(p=1.0),
                A.ImageCompression(quality_range=(65, 100), p=1.0),
            ],
            p=0.12,
        ),
        A.Sharpen(alpha=(0.10, 0.25), lightness=(0.85, 1.15), p=0.10),
    ]


model = YOLO(MODEL_CFG)
train_kwargs = dict(
    data=DATA_CFG,
    epochs=300,
    pretrained=PRETRAINED_WEIGHTS,
    imgsz=TRAIN_IMGSZ,
    batch=TRAIN_BATCH,
    project="runs/gyu_det",
    name=f"p2_imgsz{TRAIN_IMGSZ}_autobatch_bridge_aug",
    exist_ok=True,
    # Keep geometric augmentations conservative for crack-like thin defects.
    degrees=0.0,
    translate=0.06,
    scale=0.25,
    shear=0.0,
    perspective=0.0,
    # Photometric augmentations are emphasized for under-bridge scenes.
    hsv_h=0.012,
    hsv_s=0.55,
    hsv_v=0.35,
    mosaic=0.80,
    mixup=0.05,
    close_mosaic=15,
    fliplr=0.50,
    flipud=0.00,
)
custom_augmentations = build_bridge_augmentations()
if custom_augmentations is not None:
    train_kwargs["augmentations"] = custom_augmentations

results = model.train(**train_kwargs)
