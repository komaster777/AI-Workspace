import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from threading import Thread
import numpy as np
import torch
import pickle
import time
import sys
import tkinter as tk
import json
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
# from scipy.spatial.distance import cdist
from collections import deque
# import mediapipe as mp


# Настройки устройства и интерфейса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используется устройство: {device}")

# Параметры окна
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0 # Масштаб
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1920 # Ширина окна
height = int(sys.argv[3]) if len(sys.argv) > 3 else 1080 # Высота окна
output_resolution = (int(width * scale), int(height * scale)) # Разрешение окна
width_cam = int(sys.argv[4]) if len(sys.argv) > 4 else 2560 # Ширина камеры
height_cam = int(sys.argv[5]) if len(sys.argv) > 5 else 1440 # Высота камеры
resolution_cam = (int(width_cam * scale), int(height_cam * scale)) # Разрешение камеры

print(f"[INFO] Масштаб окна: {scale}, разрешение: {width}x{height}")
print(f"[INFO] Разрешение камеры: {width_cam}x{height_cam}")

# Модели

pose_model = YOLO("yolov8n-pose.pt").to(device)
pose_model.to("cuda")  # Явно переносим на GPU
object_model = YOLO("yolov8n.pt").to(device)
object_model.to("cuda")  # Явно переносим на GPU


# Проверка, работает ли YOLO pose на GPU
pose_model_device = pose_model.device
if pose_model_device.type == 'cuda':
    print(f"[INFO] YOLO pose модель работает на GPU: {pose_model_device}")
else:
    print(f"[INFO] YOLO pose модель работает на CPU")

# Порог распознавания
recognition_t = 0.6
required_size = (160, 160)


# def detect_objects_in_hands(frame, pose_results):
objects_in_hands = []



print("Запускаем камеру")

# Камера
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)
# cap.set(cv2.CAP_PROP_FPS, 30)

frame = None
ret = False

def capture_frames():
    global frame, ret
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Нет кадра с камеры в функции")
            break

# Запуск потока захвата кадров
thread = Thread(target=capture_frames, daemon=True)
thread.start()


# Тайминг
desired_fps = 30
frame_time = 1.0 / desired_fps

# Загрузка рабочей зоны из JSON
zone_path = "work_zone.json"
if os.path.exists(zone_path):
    with open(zone_path, "r") as f:
        zone_data = json.load(f)

    # Преобразуем координаты к кортежам (из списков)
    zone_bbox = [tuple(pt) for pt in zone_data.get("points", [])]
    zone_scale = zone_data.get("scale", 1.0)
    zone_resolution = zone_data.get("resolution", [1920, 1080])
    input_resolution = (int(zone_resolution[0] * zone_scale), int(zone_resolution[1] * zone_scale))
    print(f"[INFO] Координаты зоны: {zone_bbox}")
    print(f"[INFO] Масштаб зоны: {zone_scale}")
    print(f"[INFO] Разрешение зоны: {zone_resolution}")

    # if input_resolution != output_resolution:
    # Масштабируем координаты зоны под текущее разрешение и масштаб отображения
    ratio_x = (width_cam) / (zone_resolution[0] * zone_scale)
    ratio_y = (height_cam) / (zone_resolution[1] * zone_scale)
    zone_bbox = [(int(x * ratio_x), int(y * ratio_y)) for (x, y) in zone_bbox]
    print(f"[INFO] Масштабированные координаты зоны: {zone_bbox}")

else:
    # Если файл не найден — используем дефолтную зону
    print("⚠ Файл work_zone.json не найден. Используется зона по умолчанию.")
    zone_bbox = [(200, 300), (800, 300), (800, 900), (200, 900)]
    ratio_x = (width_cam) / 2560
    ratio_y = (height_cam) / 1440
    zone_bbox = [(int(x * ratio_x), int(y * ratio_y)) for (x, y) in zone_bbox]


# Конвертируем в NumPy-массив после масштабирования
zone_pts = np.array(zone_bbox, dtype=np.int32)


# Завершение по ESC
def on_escape(event=None):
    cap.release()
    cv2.destroyAllWindows()



# ЕСЛИ НУЖНО ЛОГИРОВАТЬ ЧАСТОТУ КАДРОВ В СКРИПТЕ: Remove indication.py -> inser_fps_logs
# def log_with_fps(tag, start_time):
#     elapsed = time.time() - start_time
#     fps = 1 / elapsed if elapsed > 0 else float('inf')
#     print(f"[INFO] {tag}: {fps:.3f} FPS")

camera_wait = 0
# time_count = 0

# Основной цикл
while True:
    if not ret and camera_wait >= 10:
        print("[ERROR] Нет кадра с камеры")
        break
    if not ret or frame is None:
        # Камера еще не готова, подождем 0.5 сек.
        time.sleep(0.50)
        camera_wait += 1
        print(f"[INFO] Камера не готова. Ожидание... {camera_wait} попыток")
        continue

    # Если кадр успешно получен, сбрасываем счётчик ожиданий
    camera_wait = 0
    # if time_count > 0:

    start_time = time.time()
    # time_count += 1

    # s_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)[0]

    # # Детекция объектов в руках
    # objects_in_hands = detect_objects_in_hands(frame, pose_results)
    results = object_model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    print("Вызов функции")
    if pose_results.keypoints is not None and results.boxes is not None:
        # print(f"[DEBUG] Количество поз: {len(pose_results.keypoints)}")
        # print(f"[DEBUG] Ключевые точки поз: {pose_results.keypoints}")
        # Используем атрибут 'data' для доступа к ключевым точкам
        keypoint_data = pose_results.keypoints.data  # Получаем тензор ключевых точек (shape: [N, 17, 3])
        for idx, person in enumerate(keypoint_data):
            # Получаем координаты и уверенность для ключевых точек 9 и 10 (руки)
            left_kp = person[9] if len(person) > 9 else None  # Левая рука (x, y, confidence)
            right_kp = person[10] if len(person) > 10 else None  # Правая рука (x, y, confidence)
            # print(f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}")
            # print(f"[DEBUG] Данные person: {person}, Тип: {type(person)}")
            # Координаты левой и правой руки
            if left_kp is not None and right_kp is not None and not torch.all(left_kp[:2] == 0) and not torch.all(
                    right_kp[:2] == 0):
                left_hand = left_kp[:2].cpu().numpy()
                right_hand = right_kp[:2].cpu().numpy()

                # Отрисовка зоны близости
                cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 50, (0, 255, 0),
                           5)  # желтая зона близости левой руки
                cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 50, (0, 255, 0),
                           5)  # желтая зона близости правой руки
                print("Рисуем обе")
                print(
                    f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                    distance_to_left = np.linalg.norm(left_hand - object_center)
                    distance_to_right = np.linalg.norm(right_hand - object_center)

                    if distance_to_left < 50 or distance_to_right < 50:
                        objects_in_hands.append({
                            "object": object_model.names[int(cls)],
                            "confidence": float(conf.item()),
                            "bbox": (x1, y1, x2, y2)
                        })
            else:
                # Проверка, что точки не пустые (например, не [0, 0, 0] или NaN)
                if left_kp is not None and not torch.all(left_kp[:2] == 0):
                    left_hand = left_kp[:2].cpu().numpy()

                    # Отрисовка зоны близости
                    cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 50, (0, 255, 0),
                               5)  # желтая зона близости левой руки
                    print("Рисуем левую")
                    print(
                        f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                    # Проверяем пересечение объектов с руками
                    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                        distance_to_left = np.linalg.norm(left_hand - object_center)
                        # Если объект находится близко к руке (например, < 50 пикселей)
                        if distance_to_left < 50:
                            objects_in_hands.append({
                                "object": object_model.names[int(cls)],
                                "confidence": float(conf.item()),
                                "bbox": (x1, y1, x2, y2)
                            })
                # Проверка, что точки не пустые (например, не [0, 0, 0] или NaN)
                if right_kp is not None and not torch.all(right_kp[:2] == 0):
                    right_hand = right_kp[:2].cpu().numpy()
                    cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 50, (0, 255, 0),
                               5)  # желтая зона близости правой руки
                    print("Рисуем правую")
                    print(
                        f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                    # Проверяем пересечение объектов с руками
                    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                        distance_to_right = np.linalg.norm(right_hand - object_center)

                        # Если объект находится близко к руке (например, < 50 пикселей)
                        if distance_to_right < 50:
                            objects_in_hands.append({
                                "object": object_model.names[int(cls)],
                                "confidence": float(conf.item()),
                                "bbox": (x1, y1, x2, y2)
                            })


    bodies = []
    if pose_results.boxes.id is not None:
        for box, tid in zip(pose_results.boxes.xyxy, pose_results.boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            bodies.append({'id': tid, 'bbox': (x1, y1, x2, y2)})



        face_rect_color = (0, 255, 0)  # Зеленый по умолчанию
        face_text_top = ""








        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


    # s_time = time.time()
    zone_pts = np.array(zone_bbox, dtype=np.int32)
    cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=3)

    frame_resized = cv2.resize(frame, output_resolution)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face + Body Tracking", frame_resized)

    # print(f"Время отрисовки и UI: {time.time() - s_time:.6f} секунд")

    # Проверка закрытия окна OpenCV
    if cv2.getWindowProperty("Face + Body Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
