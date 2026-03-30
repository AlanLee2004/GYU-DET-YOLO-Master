from pathlib import Path

from ultralytics import YOLO

MODEL_CFG = "ultralytics/cfg/models/master/v0_1/seg/yolo-master-seg-m.yaml"
DATA_CFG = "GZ-DET-seg.yaml"

# If one of these checkpoints exists, segmentation training will load shared detection/backbone weights first.
INIT_WEIGHT_CANDIDATES = [
    "runs/gyu_det/p2_imgsz640_autobatch/weights/best.pt",
    "runs/gyu_det/p2_imgsz960/weights/best.pt",
]
TRAIN_IMGSZ = 640  # low-VRAM fallback; try 768/960/1280 when memory allows
TRAIN_BATCH = -1  # AutoBatch: choose the largest safe batch for current GPU memory

pretrained_arg = False
for weight_path in INIT_WEIGHT_CANDIDATES:
    if Path(weight_path).exists():
        pretrained_arg = weight_path
        break
print(f"[INFO] pretrained={pretrained_arg}")

model = YOLO(MODEL_CFG)
results = model.train(
    data=DATA_CFG,
    epochs=300,
    pretrained=pretrained_arg,
    imgsz=TRAIN_IMGSZ,
    batch=TRAIN_BATCH,
    project="runs/gyu_seg",
    name=f"seg_m_imgsz{TRAIN_IMGSZ}_autobatch",
    exist_ok=True,
)
