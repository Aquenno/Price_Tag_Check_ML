from ultralytics import YOLO

# Загрузка модели (можно начать с предобученной 'yolov8n.pt')
model = YOLO('yolov8n.pt')  # или 'yolov8s.pt' для меньшей модели

# Запуск обучения
results = model.train(
    data='D:/Studying/pyataeva/configs/data.yaml',
    epochs=100,            # количество эпох
    batch=8,               # размер батча
    imgsz=640,             # размер изображения
    device='cpu'             # '0' для GPU, 'cpu' для CPU
)