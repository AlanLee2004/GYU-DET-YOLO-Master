from ultralytics import YOLO

MODEL_CFG = "ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml"
PRETRAINED_WEIGHTS = "YOLO-Master-EsMoE-M.pt"
DATA_CFG = "GZ-DET.yaml"
TRAIN_IMGSZ = 960  # switch to 1280 when GPU memory is sufficient

model = YOLO(MODEL_CFG)
results = model.train(
    data=DATA_CFG,
    epochs=300,
    pretrained=PRETRAINED_WEIGHTS,
    imgsz=TRAIN_IMGSZ,
    project="runs/gyu_det",
    name=f"p2_imgsz{TRAIN_IMGSZ}",
    exist_ok=True,
)
