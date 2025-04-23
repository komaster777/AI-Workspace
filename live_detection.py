import cv2
import torch
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from train_model import PoseLSTM  # Импортируем класс модели

# --- Создание модели и загрузка состояния ---
model = PoseLSTM()  # Создаем экземпляр модели
checkpoint = torch.load("best_model.pth", map_location=torch.device("cuda"))  # Загружаем сохранённую модель
model.load_state_dict(checkpoint['model_state_dict'])  # Загружаем веса модели
model.eval()  # Переводим модель в режим оценки (не обучаем)

# Инициализация Mediapipe и YOLO
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
yolo = YOLO("yolov8n.pt")

# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Перемещаем модель на GPU, если доступно
yolo.to(device)

# Выводим информацию о том, какой девайс используется
print(f"YOLO использует {device}")

# Открытие видео с разрешением
cap = cv2.VideoCapture(0)

# Установка разрешения видео
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ширина
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Высота

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Получаем позу (Mediapipe) ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    pose_landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.extend([landmark.x, landmark.y, landmark.z])

    if len(pose_landmarks) != 99:
        continue

    # --- Определяем книгу (YOLOv8) ---
    results_yolo = yolo(frame)
    book_detected = 0
    for result in results_yolo:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 73:
                book_detected = 1

    # --- Предсказание модели ---
    input_tensor = torch.tensor([pose_landmarks + [book_detected]], dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        prediction = model(input_tensor)
        pred_label = torch.argmax(prediction, dim=1).item()

    # --- Отображение результата ---
    text = "READING" if pred_label == 1 else "NOT READING"
    color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Reading Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
