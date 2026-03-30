from pathlib import Path

from ultralytics import YOLO

MODEL_CFG = "ultralytics/cfg/models/master/v0_1/seg/yolo-master-seg-m.yaml"
DATA_CFG = "GZ-DET-seg.yaml"

# If this checkpoint exists, segmentation training will load shared detection/backbone weights first.
INIT_WEIGHTS = "runs/gyu_det/p2_imgsz960/weights/best.pt"
TRAIN_IMGSZ = 960  # switch to 1280 if GPU memory is sufficient

pretrained_arg = INIT_WEIGHTS if Path(INIT_WEIGHTS).exists() else False
print(f"[INFO] pretrained={pretrained_arg}")

model = YOLO(MODEL_CFG)
results = model.train(
    data=DATA_CFG,
    epochs=300,
    pretrained=pretrained_arg,
    imgsz=TRAIN_IMGSZ,
    project="runs/gyu_seg",
    name=f"seg_m_imgsz{TRAIN_IMGSZ}",
    exist_ok=True,
)
