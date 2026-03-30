from ultralytics import YOLO

MODEL_CFG = "ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml"
PRETRAINED_WEIGHTS = "YOLO-Master-EsMoE-M.pt"
DATA_CFG = "GZ-DET.yaml"
TRAIN_IMGSZ = 640  # low-VRAM fallback; try 768/960/1280 when memory allows
TRAIN_BATCH = -1  # AutoBatch: choose the largest safe batch for current GPU memory

model = YOLO(MODEL_CFG)
results = model.train(
    data=DATA_CFG,
    epochs=300,
    pretrained=PRETRAINED_WEIGHTS,
    imgsz=TRAIN_IMGSZ,
    batch=TRAIN_BATCH,
    project="runs/gyu_det",
    name=f"p2_imgsz{TRAIN_IMGSZ}_autobatch",
    exist_ok=True,
)
