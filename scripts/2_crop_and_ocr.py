import cv2
import easyocr
from ultralytics import YOLO

# 1. Детекция ценника
model = YOLO("models/yolo/trained/best.pt")
img = cv2.imread("data/raw/image1.jpg")
results = model(img)
bbox = results[0].boxes.xyxy[0].cpu().numpy()  # Координаты bbox

# 2. Обрезка ценника
x1, y1, x2, y2 = map(int, bbox)
cropped = img[y1:y2, x1:x2]

# 3. Распознавание текста
reader = easyocr.Reader(["ru"])
text = reader.readtext(cropped, detail=0)  # ['Молоко', '99.99 руб']
print(text)