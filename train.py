from ultralytics import YOLO

model = YOLO("YOLO-Master-EsMoE-M.pt")

results = model.train(
    data="GZ-DET.yaml",
    epochs=300,
    pretrained=True
)