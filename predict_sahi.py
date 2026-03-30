from ultralytics import YOLO

MODEL_WEIGHTS = "runs/gyu_det/p2_imgsz960/weights/best.pt"
SOURCE = "data/test_images"

# Keep inference at high resolution; tune to 1280 if GPU memory allows.
PREDICT_IMGSZ = 1280
SLICE_SIZE = 960
OVERLAP_RATIO = 0.2

model = YOLO(MODEL_WEIGHTS)
results = model.predict(
    source=SOURCE,
    imgsz=PREDICT_IMGSZ,
    conf=0.25,
    iou=0.60,
    sparse_sahi=True,
    slice_size=SLICE_SIZE,
    overlap_ratio=OVERLAP_RATIO,
    save=True,
    project="runs/gyu_det",
    name=f"sparse_sahi_img{PREDICT_IMGSZ}_slice{SLICE_SIZE}",
    exist_ok=True,
)
