from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Предобученная модель
results = model.train(
    data="configs/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32
)