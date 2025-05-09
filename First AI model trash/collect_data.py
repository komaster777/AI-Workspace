import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO

# Инициализация Mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Инициализация YOLOv8
yolo = YOLO("yolov8n.pt")  # Используем предобученную YOLOv8

# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Перемещаем модель на GPU, если доступно
yolo.to(device)

# Выводим информацию о том, какой девайс используется
print(f"YOLO использует {device}")


# Файл для сохранения данных
CSV_FILE = "pose_data.csv"
columns = [f"x{i}" for i in range(33)] + [f"y{i}" for i in range(33)] + [f"z{i}" for i in range(33)] + ["book_in_hand", "reading"]
data = []

cap = cv2.VideoCapture(0)

# Установка разрешения видео
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

print("Начинаем сбор данных... Держите книгу и читайте.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр.")
        break

    # --- Mediapipe 3D Pose ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    pose_landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.extend([landmark.x, landmark.y, landmark.z])

    if len(pose_landmarks) != 99:
        continue  # Если поза не найдена, пропускаем кадр

    # --- YOLOv8 (Поиск книги) ---
    results_yolo = yolo(frame)
    book_detected = 0
    for result in results_yolo:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 73:  # Класс книги в COCO
                book_detected = 1

    # --- Ввод метки ---
    reading = 0
    # Используем клавишу для ввода
    print("Нажмите '1' если читаете книгу, '0' если нет.")
    key = cv2.waitKey(1) & 0xFF  # Увеличено время ожидания
    if key == ord('1'):
        reading = 1
    elif key == ord('0'):
        reading = 0
    elif key == 27:  # ESC для выхода
        break

    # Сохранение данных
    data.append(pose_landmarks + [book_detected, reading])

    # Отображение видео
    cv2.putText(frame, "Book Detected" if book_detected else "No Book", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if book_detected else (0, 0, 255), 2)
    cv2.imshow("Data Collection", frame)

cap.release()
cv2.destroyAllWindows()

# Запись в CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(CSV_FILE, index=False)
print(f"Данные сохранены в {CSV_FILE}")