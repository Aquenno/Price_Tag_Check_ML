import cv2
import easyocr
from ultralytics import YOLO
import gradio as gr

# Инициализация моделей
model = YOLO("C:/Users/user/Desktop/Price_Tag_Check_ML/runs/detect/train6/weights/best.pt")
reader = easyocr.Reader(["ru"])

def process_image(img):
    # Конвертация цветового пространства
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Словари для хранения результатов
    class_data = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    # Детекция объектов
    results = model(img_bgr)
    
    # Обработка найденных объектов
    for box in results[0].boxes:
        # Получение параметров bbox
        bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        cls_id = int(box.cls)
        
        # Обрезка и распознавание текста
        cropped = img_bgr[y1:y2, x1:x2]
        text = reader.readtext(cropped, detail=0)
        combined_text = ' '.join(text).strip()
        
        if cls_id in class_data:
            class_data[cls_id].append((x1, combined_text))
    
    # Сортировка текстов по координате X
    sorted_texts = {}
    for cls_id in class_data:
        sorted_texts[cls_id] = [text for _, text in sorted(class_data[cls_id], key=lambda x: x[0])]
    
    # Формирование результата
    description = ' '.join(sorted_texts[0]) if sorted_texts[0] else "Не обнаружено"
    
    price_normal = ', '.join(sorted_texts[3] + sorted_texts[1]) 
    if not price_normal: 
        price_normal = "Не обнаружено"
        
    price_card = ', '.join(sorted_texts[4] + sorted_texts[2])
    if not price_card:
        price_card = "Не обнаружено"

    return description, price_normal, price_card

# Создание интерфейса
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Загрузите изображение ценника"),
    outputs=[
        gr.Textbox(label="Описание продукта", placeholder="Описание..."),
        gr.Textbox(label="Цена без карты", placeholder="Цена..."),
        gr.Textbox(label="Цена по карте", placeholder="Цена...")
    ],
    title="📷 CV для распознавания ценников на полках",
    description="Загрузите фото ценника для распознавания текста",
    allow_flagging="never"
)

# Запуск приложения
if __name__ == "__main__":
    interface.launch(share=False)