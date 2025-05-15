import cv2
import easyocr
from ultralytics import YOLO

# Инициализация модели YOLO и EasyOCR
model = YOLO("C:/Users/user/Desktop/Price_Tag_Check_ML/runs/detect/train6/weights/best.pt")
reader = easyocr.Reader(["ru"])

# Загрузка изображения
img = cv2.imread("C:/Users/user/Desktop/Price_Tag_Check_ML/data/raw/image6.jpg")

# Детекция объектов
results = model(img)

# Перебор всех обнаруженных объектов
for i, box in enumerate(results[0].boxes):
    # Получение координат bbox и класса
    bbox = box.xyxy[0].cpu().numpy()
    cls_id = int(box.cls)
    conf = float(box.conf)
    
    # Обрезка объекта
    x1, y1, x2, y2 = map(int, bbox)
    cropped = img[y1:y2, x1:x2]
    
    # Сохранение обрезанного изображения (для отладки)
    cv2.imwrite(f"cropped_object_{i}.jpg", cropped)
    
    # Распознавание текста
    text = reader.readtext(cropped, detail=0)
    
    # Вывод информации
    print(f"Объект {i}:")
    print(f"  Класс: {cls_id} (Уверенность: {conf:.2f})")
    print(f"  Координаты: [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Распознанный текст:", " ".join(text))
    print("-" * 50)

# Визуализация результатов (если нужно)
results_plotted = results[0].plot()  # Рисует bbox и подписи на изображении
cv2.imwrite("detected_objects.jpg", results_plotted)
print("Результаты детекции сохранены в detected_objects.jpg")