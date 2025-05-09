import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import array
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
from insightface.app import FaceAnalysis
import mediapipe as mp
# from scipy.spatial.distance import cdist
from collections import deque
#nvcc --version
#pipreqs . --force
#pip freeze > requirements_all.txt


# Настройки устройства и интерфейса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используется устройство: {device}")

# Окно Tkinter
root = tk.Tk()
root.title("Присутствие сотрудников")
root.geometry("400x600") # размеры окна ТКинтер
status_frame = ttk.Frame(root)
status_frame.pack(fill=tk.BOTH, expand=True)
status_labels = {}

# Цветные иконки статуса
def get_status_icon(present=True):
    size = 20
    img = Image.new("RGB", (size, size), (0, 255, 0) if present else (255, 0, 0))
    if present:
        for x in range(size):
            for y in range(size):
                if (x - size // 2) ** 2 + (y - size // 2) ** 2 > (size // 2) ** 2:
                    img.putpixel((x, y), (255, 255, 255))
    return ImageTk.PhotoImage(img)

status_icons = {
    True: get_status_icon(True),
    False: get_status_icon(False),
}

# Создание строки в UI
def create_status_row(name):
    row = ttk.Frame(status_frame)
    row.pack(fill=tk.X, pady=2)
    icon_label = tk.Label(row, image=status_icons[False])
    icon_label.pack(side=tk.LEFT, padx=5)
    name_label = ttk.Label(row, text=f"{name}: ⬛ Не на месте", anchor="w")
    name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    status_labels[name] = {
        "row": row,
        "icon": icon_label,
        "label": name_label
    }


# Обновление строки в UI
def update_status_row(name, present, objects_in_hands):
    if name == "Recognizing...":
        # Не обновляем статус для "Распознается..."
        return
    if name not in status_labels:
        create_status_row(name)
    if objects_in_hands:
        object_names = ", ".join([f"{obj['object']} ({obj['confidence']:.2f})" for obj in objects_in_hands])
        text = f"\U0001F7E2 На рабочем месте: {object_names}" if present else f"⬛ Не на месте: {object_names}"
    else:
        text = "\U0001F7E2 На рабочем месте" if present else "⬛ Не на месте"
    icon = status_icons[present]
    status_labels[name]["icon"].config(image=icon)
    status_labels[name]["label"].config(text=f"{name}: {text}")

# Обновление строки с информацией о предметах
# def update_status_with_objects(name, objects_in_hands):
#     if name not in status_labels:
#         create_status_row(name)
#
#     if objects_in_hands:
#         object_names = ", ".join([f"{obj['object']} ({obj['confidence']:.2f})" for obj in objects_in_hands])
#         text = f"\U0001F7E2 На рабочем месте: {object_names}"
#     else:
#         text = "\U0001F7E2 На рабочем месте (без предметов)"
#
#     status_labels[name]["label"].config(text=f"{name}: {text}")

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
object_model = YOLO("yolo11s.pt").to(device)
object_model.to("cuda")  # Явно переносим на GPU
pose_model = YOLO("yolov8n-pose.pt").to(device)
pose_model.to("cuda")  # Явно переносим на GPU
# Загрузка модели InsightFace
face_app = FaceAnalysis(name='buffalo_l')
ctx_id = 0 if torch.cuda.is_available() else -1
face_app.prepare(ctx_id=ctx_id)
# Проверка провайдеров
print("[INFO] Провайдеры модели:", face_app.models['recognition'].session.get_providers())
# # Инициализация модели отслеживания рук с использованием GPU
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# Проверка, работает ли YOLO на GPU
object_model_device = object_model.device
if object_model_device.type == 'cuda':
    print(f"[INFO] YOLO модель работает на GPU: {object_model_device}")
else:
    print(f"[INFO] YOLO модель работает на CPU")
# Проверка, работает ли YOLO pose на GPU
pose_model_device = pose_model.device
if pose_model_device.type == 'cuda':
    print(f"[INFO] YOLO pose модель работает на GPU: {pose_model_device}")
else:
    print(f"[INFO] YOLO pose модель работает на CPU")


# Порог распознавания
recognition_t = 0.6 # Порог распознания
required_size = (160, 160)
hands_r = 300 # Радиус рук (порог)

# Функция загрузки эмбеддингов лиц
def load_encodings(path="face_encodings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["encodings"]), data["names"]

# Загрузка известных лиц
known_face_encodings_np, known_face_names = load_encodings()
# Нормализуем эмбеддинги
known_norm = known_face_encodings_np / np.linalg.norm(known_face_encodings_np, axis=1, keepdims=True)

# Функция для детекции и извлечения эмбеддингов лиц
def detect_and_encode_faces(image):
    faces = face_app.get(image)
    face_encodings = []
    face_locations = []
    face_confidences = []  # Для хранения уверенности детекции

    for face in faces:
        embedding = face.normed_embedding  # Нормализованный эмбеддинг
        bbox = face.bbox.astype(int)  # Прямоугольник лица
        confidence = face.det_score  # Уверенность детекции лица
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        face_encodings.append(embedding)
        face_locations.append((x1, y1, w, h))
        face_confidences.append(confidence)  # Сохраняем уверенность

    return face_locations, face_encodings, face_confidences

# Распознавание лиц
def recognize_faces(face_encodings):
    recognized_faces = []

    for encoding in face_encodings:
        # Нормализуем входной эмбеддинг
        encoding_norm = encoding / np.linalg.norm(encoding)
        # Вычисляем косинусное расстояние
        distances = 1 - np.dot(known_norm, encoding_norm)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        # Распознаем лицо, если расстояние меньше порога
        name = known_face_names[min_dist_idx] if min_dist < recognition_t else "Unknown"
        recognized_faces.append((name, min_dist, min_dist_idx))  # Возвращаем индекс

    return recognized_faces


def draw_half_circle(frame, hand_coords, elbow_coords, radius, color=(0, 255, 0), thickness=5):
    """
    Рисует полуокружность от руки (hand_coords) в направлении локтя (elbow_coords).

    :param frame: Кадр изображения.
    :param hand_coords: Координаты кисти (x, y).
    :param elbow_coords: Координаты локтя (x, y).
    :param radius: Радиус окружности.
    :param color: Цвет дуги.
    :param thickness: Толщина дуги.
    """
    # Координаты кисти
    hand_x, hand_y = hand_coords
    # Координаты локтя
    elbow_x, elbow_y = elbow_coords

    # Вычисляем угол направления (в градусах)
    angle = np.arctan2(hand_y - elbow_y, hand_x - elbow_x) * 180 / np.pi

    vector_x = hand_x - elbow_x
    vector_y = hand_y - elbow_y

    length = np.sqrt(vector_x ** 2 + vector_y ** 2)
    if length != 0:  # Чтобы не делить на 0
        vector_x /= length
        vector_y /= length

    # Определяем углы для рисования полуокружности
    start_angle = angle - 90  # Начало дуги
    end_angle = angle + 90  # Конец дуги

    scale = 100  # Длина, на которую нужно изменить координаты
    hand_x = hand_x + vector_x * scale
    hand_y = hand_y + vector_y * scale
    print(f"x: {hand_x}, y: {hand_y}")

    # Рисуем полуокружность
    cv2.ellipse(
        frame,
        (int(hand_x), int(hand_y)),  # Центр дуги
        (int(radius), int(radius)),  # Радиус
        0,  # Наклон эллипса (0 для круга)
        start_angle,  # Угол начала
        end_angle,  # Угол конца
        color,  # Цвет
        thickness  # Толщина линии
    )
    # Вычисляем координаты концов дуги
    start_x = int(hand_x + radius * np.cos(np.radians(start_angle)))
    start_y = int(hand_y + radius * np.sin(np.radians(start_angle)))
    end_x = int(hand_x + radius * np.cos(np.radians(end_angle)))
    end_y = int(hand_y + radius * np.sin(np.radians(end_angle)))

    # Рисуем линию, соединяющую два конца дуги
    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

# # Функция для вычисления пересечения двух прямоугольников
# def rects_intersect(body_rect, obj_rect):
#     x1_body, y1_body, x2_body, y2_body = body_rect
#     x1_obj, y1_obj, x2_obj, y2_obj = obj_rect
#
#     # Вычисляем координаты пересечения
#     x_left = max(x1_body, x1_obj)
#     y_top = max(y1_body, y1_obj)
#     x_right = min(x2_body, x2_obj)
#     y_bottom = min(y2_body, y2_obj)
#
#     # Проверяем, есть ли пересечение
#     if x_left < x_right and y_top < y_bottom:
#         return True  # Прямоугольники пересекаются
#     return False

# # Список классов, которые нужно игнорировать
#     ignored_classes = ["person", "car", "cat"]
#
#     if object_results.boxes is not None and len(object_results.boxes) > 0:
#         for box, conf, cls in zip(object_results.boxes.xyxy, object_results.boxes.conf,
#                                   object_results.boxes.cls):
#             class_name = object_model.names[int(cls)]
#             if class_name not in ignored_classes:
#                 x1, y1, x2, y2 = map(int, box.tolist())
#                 # Рисуем прямоугольник вокруг объекта
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 label = f"{object_model.names[int(cls)]} ({conf:.2f})"
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def detect_objects_in_hands(frame, pose_results):
    objects_in_hands = []
    key_hand = array.array('i')
    ignored_classes = ["person", "car", "cat"]
    base_width = 1920
    base_height = 1080
    base_width_cam = 2560
    base_height_cam = 1440
    distance_threshold = hands_r

    scale_factor = min(width, height) / min(base_width, base_height)
    scale_factor_cam = min(width_cam, height_cam) / min(base_width_cam, base_height_cam)
    distance_threshold = distance_threshold * scale_factor
    distance_threshold = distance_threshold * scale_factor_cam



    results = object_model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    # print("Вызов функции")
    if pose_results.keypoints is not None and results.boxes is not None:
        # print(f"[DEBUG] Количество поз: {len(pose_results.keypoints)}")
        # print(f"[DEBUG] Ключевые точки поз: {pose_results.keypoints}")
        # Используем атрибут 'data' для доступа к ключевым точкам
        keypoint_data = pose_results.keypoints.data  # Получаем тензор ключевых точек (shape: [N, 17, 3])
        for idx, person in enumerate(keypoint_data):
            # Получаем координаты и уверенность для ключевых точек 9 и 10 (руки)
            left_kp = person[9] if len(person) > 9 else None  # Левая рука (x, y, confidence)
            right_kp = person[10] if len(person) > 10 else None  # Правая рука (x, y, confidence)
            left_eb = person[7] if len(person) > 7 else None  # лев локоть
            right_eb = person[8] if len(person) > 8 else None  # Прав локоть
            # print(f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}")
            # print(f"[DEBUG] Данные person: {person}, Тип: {type(person)}")
            # Координаты левой и правой руки
            if (left_kp is not None and right_kp is not None and left_eb is not None and right_eb is not None and
                    not torch.all(left_kp[:2] == 0) and not torch.all(right_kp[:2] == 0) and not torch.all(left_eb[:2] == 0) and not torch.all(right_eb[:2] == 0)):
                left_hand = left_kp[:2].cpu().numpy()
                right_hand = right_kp[:2].cpu().numpy()
                left_elbow = left_eb[:2].cpu().numpy()
                right_elbow = right_eb[:2].cpu().numpy()

                # Отрисовка зоны близости
                # cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 50, (0, 255, 0),
                #            5)  # желтая зона близости левой руки
                # cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 50, (0, 255, 0),
                #            5)  # желтая зона близости правой руки
                key_hand.extend([1, int(left_hand[0]), int(left_hand[1]), int(left_elbow[0]), int(left_elbow[1]),
                                 int(right_hand[0]), int(right_hand[1]), int(right_elbow[0]), int(right_elbow[1])])
                # print("Рисуем обе")
                # print(f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                    class_name = object_model.names[int(cls)]
                    if class_name not in ignored_classes:
                        x1, y1, x2, y2 = map(int, box.tolist())

                        object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                        distance_to_left = np.linalg.norm(left_hand - object_center)
                        distance_to_right = np.linalg.norm(right_hand - object_center)

                        # # Вычисляем ширину и высоту объекта
                        # obj_width = x2 - x1
                        # obj_height = y2 - y1
                        #
                        # # Динамический порог на основе размера объекта
                        # distance_threshold = distance_threshold * max(obj_width,
                        #                                               obj_height) * 1.5  # Увеличиваем порог на 50% от размера объекта

                        if distance_to_left < distance_threshold or distance_to_right < distance_threshold:
                            objects_in_hands.append({
                                "object": object_model.names[int(cls)],
                                "confidence": float(conf.item()),
                                "bbox": (x1, y1, x2, y2)
                            })

                            # Отрисовка прямоугольника
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Прямоугольник
                            label = f"{object_model.names[int(cls)]} ({conf:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Проверка, что точки не пустые (например, не [0, 0, 0] или NaN)
                if left_kp is not None and not torch.all(left_kp[:2] == 0) and left_eb is not None and not torch.all(left_eb[:2] == 0):
                    left_hand = left_kp[:2].cpu().numpy()
                    left_elbow = left_eb[:2].cpu().numpy()
                    # Отрисовка зоны близости
                    # cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 50, (0, 255, 0),
                    #            5)  # желтая зона близости левой руки
                    key_hand.extend([1, int(left_hand[0]), int(left_hand[1]), int(left_elbow[0]), int(left_elbow[1]), 0, 0, 0, 0])
                    # print("Рисуем левую")
                    # print(f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                    # Проверяем пересечение объектов с руками
                    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                        class_name = object_model.names[int(cls)]
                        if class_name not in ignored_classes:
                            x1, y1, x2, y2 = map(int, box.tolist())
                            object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                            distance_to_left = np.linalg.norm(left_hand - object_center)

                            # # Вычисляем ширину и высоту объекта
                            # obj_width = x2 - x1
                            # obj_height = y2 - y1

                            # # Динамический порог на основе размера объекта
                            # distance_threshold = distance_threshold * max(obj_width,
                            #                          obj_height) * 1.5  # Увеличиваем порог на 50% от размера объекта

                            # Если объект находится близко к руке (например, < 50 пикселей)
                            if distance_to_left < distance_threshold:
                                objects_in_hands.append({
                                    "object": object_model.names[int(cls)],
                                    "confidence": float(conf.item()),
                                    "bbox": (x1, y1, x2, y2)
                                })

                                # Отрисовка прямоугольника
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Прямоугольник
                                label = f"{object_model.names[int(cls)]} ({conf:.2f})"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Проверка, что точки не пустые (например, не [0, 0, 0] или NaN)
                if right_kp is not None and not torch.all(right_kp[:2] == 0) and right_eb is not None and not torch.all(right_eb[:2] == 0):
                    right_hand = right_kp[:2].cpu().numpy()
                    right_elbow = right_eb[:2].cpu().numpy()
                    # cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 50, (0, 255, 0),
                    #            5)  # желтая зона близости правой руки
                    key_hand.extend([1, 0, 0, 0, 0, int(right_hand[0]), int(right_hand[1]), int(right_elbow[0]), int(right_elbow[1])])
                    # print("Рисуем правую")
                    # print(f"[DEBUG] Левая рука: {left_kp}, Правая рука: {right_kp}, уверенность правой{right_kp[2]}, уверенность левой {left_kp[2]}")
                    # Проверяем пересечение объектов с руками
                    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                        class_name = object_model.names[int(cls)]
                        if class_name not in ignored_classes:
                            x1, y1, x2, y2 = map(int, box.tolist())
                            object_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                            distance_to_right = np.linalg.norm(right_hand - object_center)
                            # # Вычисляем ширину и высоту объекта
                            # obj_width = x2 - x1
                            # obj_height = y2 - y1

                            # # Динамический порог на основе размера объекта
                            # distance_threshold = distance_threshold * max(obj_width,
                            #                                               obj_height) * 1.5  # Увеличиваем порог на 50% от размера объекта

                            # Если объект находится близко к руке (например, < 50 пикселей)
                            if distance_to_right < distance_threshold:
                                objects_in_hands.append({
                                    "object": object_model.names[int(cls)],
                                    "confidence": float(conf.item()),
                                    "bbox": (x1, y1, x2, y2)
                                })

                                # Отрисовка прямоугольника
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Прямоугольник
                                label = f"{object_model.names[int(cls)]} ({conf:.2f})"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return objects_in_hands, key_hand, distance_threshold

# Проверка пересечения прямоугольника и четырехугольника
def bbox_intersects_polygon(bbox, polygon):
    x1, y1, x2, y2 = bbox
    corners = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.int32)
    intersection, _ = cv2.intersectConvexConvex(corners, np.array(polygon, dtype=np.int32))
    return intersection > 0



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
    zone_resol_cam = zone_data.get("resolution_cam", [2560, 1440])
    input_resolution = (int(zone_resolution[0] * zone_scale), int(zone_resolution[1] * zone_scale))
    print(f"[INFO] Координаты зоны: {zone_bbox}")
    print(f"[INFO] Масштаб зоны: {zone_scale}")
    print(f"[INFO] Разрешение зоны: {zone_resolution}")
    print(f"[INFO] Разрешение камеры зоны: {zone_resol_cam}")

    # if input_resolution != output_resolution:
    # Масштабируем координаты зоны под текущее разрешение и масштаб отображения
    ratio_x = (width_cam) / (zone_resolution[0] * zone_scale)
    ratio_y = (height_cam) / (zone_resolution[1] * zone_scale)
    resolution_cam_new = ((width_cam), (height_cam)) # Разрешение камеры новое
    # zone_bbox = [(int(x * ratio_x), int(y * ratio_y)) for (x, y) in zone_bbox]
    # print(f"[INFO] Масштабированные координаты зоны: {zone_bbox}")

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

# Отслеживаемые люди: track_id → {name, bbox, last_seen, first_seen}
tracked_people = {}
# Временные лица в статусе "Recognizing..."
pending_faces = {}

# Завершение по ESC
def on_escape(event=None):
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

root.bind("<Escape>", on_escape)

# ЕСЛИ НУЖНО ЛОГИРОВАТЬ ЧАСТОТУ КАДРОВ В СКРИПТЕ: Remove indication.py -> inser_fps_logs
# def log_with_fps(tag, start_time):
#     elapsed = time.time() - start_time
#     fps = 1 / elapsed if elapsed > 0 else float('inf')
#     print(f"[INFO] {tag}: {fps:.3f} FPS")

camera_wait = 0

# Основной цикл
while True:
    if not ret and camera_wait >= 100:
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

    start_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations, face_encodings, confidence = detect_and_encode_faces(frame_rgb)
    recognized_faces = recognize_faces(face_encodings)

    pose_results = pose_model.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)[0]

    # Обрабатываем кадр для получения ключевых точек рук MediaPipe
    # hand_results = hands.process(frame_rgb)

    # Детекция объектов в руках
    objects_in_hands, key_hand, distance_threshold = detect_objects_in_hands(frame, pose_results)
    # key_hand.extend([1, left_hand[0], left_hand[1], right_hand[0], right_hand[1]])
    # Ограничиваем значение distance_threshold
    distance_threshold = min(distance_threshold, min(width, height) // 4)  # Не больше 1/4 меньшей стороны


    # Радиус полуокружности
    radius = int(distance_threshold)

    if len(key_hand) >= 9:
        # Пример: рисование полуокружностей для рук
        left_hand = (int(key_hand[1]), int(key_hand[2]))
        right_hand = (int(key_hand[5]), int(key_hand[6]))
        left_elbow = (int(key_hand[3]), int(key_hand[4]))  # Координаты локтя для левой руки
        right_elbow = (int(key_hand[7]), int(key_hand[8]))  # Координаты локтя для правой руки
        if left_hand != (0, 0) and left_elbow != (0, 0):
            draw_half_circle(frame, left_hand, left_elbow, radius, color=(0, 255, 0), thickness=5)
            # cv2.circle(frame, (key_hand[1], key_hand[2]), int(distance_threshold), (0, 255, 0),
            #            5)  # желтая зона близости правой руки
        if right_hand != (0, 0) and right_elbow != (0, 0):
            draw_half_circle(frame, right_hand, right_elbow, radius, color=(0, 255, 0), thickness=5)
            # cv2.circle(frame, (key_hand[3], key_hand[4]), int(distance_threshold), (0, 255, 0),
            #            5)  # желтая зона близости правой руки


    bodies = []
    if pose_results.boxes.id is not None:
        for box, tid in zip(pose_results.boxes.xyxy, pose_results.boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            bodies.append({'id': tid, 'bbox': (x1, y1, x2, y2)})
    # print(f"Время работы с box: {time.time() - s_time:.6f} секунд")
    for (x, y, w, h), confidence_detect, (name, min_dist, min_dist_idx) in zip(face_locations, confidence, recognized_faces):
        face_img = frame[y:y + h, x:x + w]
        if face_img is None or face_img.size == 0:
            continue


        confidence = 1 - min_dist



        face_center = np.array([x + w / 2, y + h / 2])
        assigned_body, min_body_dist = None, float("inf")

        for body in bodies:
            bx1, by1, bx2, by2 = body['bbox']
            center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
            dist = np.linalg.norm(center - face_center)
            if dist < min_body_dist:
                assigned_body, min_body_dist = body, dist

        face_rect_color = (0, 255, 0)  # Зеленый по умолчанию
        face_text_top = ""

        if assigned_body:
            track_id = assigned_body['id']



            if track_id in tracked_people:
                now = time.time()
                current_name = tracked_people[track_id]['name'] # Старое имя

                # ✅ Обновление face_bbox
                # if name == "Unknown":
                tracked_people[track_id]['face_bbox'] = (x, y, w, h)

                # Проверяем, что лицо присутствует
                if face_img is not None and face_img.size > 0:

                    # Перепроверка лиц на точность распознания
                    if now - tracked_people[track_id].get("last_check2", 0) >= 1.0:
                        # print("[INFO] Проверка на перераспознание")
                        # print(f"[INFO] min_dist: {min_dist}")
                        if min_dist < recognition_t:
                            new_name = known_face_names[min_dist_idx.item()]

                            # Проверяем, изменилось ли имя
                            if new_name != current_name and new_name != "Unknown":
                                tracked_people[track_id]['name'] = new_name  # Назначаем новое имя
                                tracked_people[track_id]['confidence_window'].clear()  # Очищаем окно уверенности
                                # print(f"[INFO] Очередь confidence_window для ID {track_id} очищена.")
                                cv2.imwrite(f"debug_faces/Перезапись{new_name}_{int(time.time())}.jpg", face_img)
                                print(f"[INFO] Новое имя: {new_name}")
                                still_present = any(
                                    p['name'] == current_name and bbox_intersects_polygon(p['bbox'], zone_bbox)
                                    for track_id2, p in tracked_people.items() if track_id2 != track_id
                                )
                                if not still_present and current_name in status_labels:
                                    update_status_row(current_name, False, objects_in_hands)
                                    if current_name == "Unknown":
                                        status_labels[current_name]["row"].destroy()
                                        del status_labels[current_name]

                        # Обновляем время последней проверки
                        tracked_people[track_id]['last_check2'] = now

                    # Сброс в Unknown при низкой уверенности в распозновании
                    if now - tracked_people[track_id].get("last_check1", 0) >= 0.4 and name != "Unknown":
                        # Получаем координаты лица
                        fx, fy, fw, fh = tracked_people[track_id].get('face_bbox', (0, 0, 0, 0))
                        face_img = frame[fy:fy + fh, fx:fx + fw]


                        conf = 1.0 - min_dist  # Уверенность
                        is_confident = conf > 0.3  # Уверенность, что распознанно верно
                        tracked_people[track_id]['confidence_window'].append(is_confident)  # Добавление в конце очереди словаря True/False

                        print(f"[INFO] Старое имя: {current_name}")
                        print(f"[INFO] Уверенность: {conf} ")
                        print(f"[INFO] Очередь: {tracked_people[track_id].get('confidence_window')} ")

                        # Если более 4 из 5 проверок — низкая уверенность
                        if tracked_people[track_id]['confidence_window'].count(False) >= 4:
                            if current_name != "Unknown":
                                tracked_people[track_id]['name'] = "Unknown"
                                still_present = any(
                                    p['name'] == current_name and bbox_intersects_polygon(p['bbox'], zone_bbox)
                                    for track_id2, p in tracked_people.items() if track_id2 != track_id
                                )
                                if not still_present and current_name in status_labels:
                                    update_status_row(current_name, False, objects_in_hands)

                        # Обновляем время последней проверки
                        tracked_people[track_id]['last_check1'] = now

                name = tracked_people[track_id]['name']

                recog_conf = 1.0 - min_dist
                face_text_top = f"ID: {track_id} - {name} ({recog_conf:.2f})"
                face_rect_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            elif track_id not in tracked_people:
                now = time.time()

                if track_id not in pending_faces:
                    pending_faces[track_id] = {
                        'encoding': face_encodings,
                        'first_seen': now,
                        'bbox': assigned_body['bbox'],
                        'face_bbox': (x, y, w, h)
                    }
                else:
                    # ✅ обновляем координаты лица каждый кадр, пока лицо распознается
                    pending_faces[track_id]['face_bbox'] = (x, y, w, h)
                    if min_dist < recognition_t:
                        # used_names = [p['name'] for p in tracked_people.values()]
                        matched_name = known_face_names[min_dist_idx.item()]
                        # if matched_name not in used_names:
                        tracked_people[track_id] = {
                            'name': matched_name,
                            'bbox': assigned_body['bbox'],
                            'face_bbox': (x, y, w, h),
                            'last_seen': now,
                            'first_seen': now,
                            'last_retry': now,
                            'last_check1': now,
                            'last_check2': now,
                            'last_objects': [],
                            'IDs': 0,
                            'confidence_window': deque(maxlen=5)
                        }
                        del pending_faces[track_id]
                    elif now - pending_faces[track_id]['first_seen'] >= 3.0:
                        matched_name = "Unknown"
                        tracked_people[track_id] = {
                            'name': matched_name,
                            'bbox': assigned_body['bbox'],
                            'face_bbox': (x, y, w, h),
                            'last_seen': now,
                            'first_seen': now,
                            'last_retry': now,
                            'last_check1': now,
                            'last_check2': now,
                            'last_objects': [],
                            'confidence_window': deque(maxlen=5)
                        }
                        del pending_faces[track_id]

            if track_id in pending_faces:
                    face_text_top = f"ID: {track_id} - Recognizing..."
                    face_rect_color = (0, 255, 255)

        # --- Отрисовка прямоугольника вокруг лица и текстов ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_rect_color, 2)

        if face_text_top:
            cv2.putText(frame, face_text_top, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_rect_color, 2)

        cv2.putText(frame, f"{confidence_detect:.2f}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for body in bodies:
        if body['id'] not in tracked_people:
            continue
        track_id = body['id']
        if track_id in tracked_people:
            tracked_people[track_id]['bbox'] = body['bbox']
            tracked_people[track_id]['last_seen'] = time.time()

    # удаления записей о людях из списка tracked_people, если они не были замечены в кадре более 5 секунд
    for tid in list(set(list(tracked_people.keys()) + list(pending_faces.keys()))):
        if tid in tracked_people and time.time() - tracked_people[tid]['last_seen'] > 5:
            name = tracked_people[tid]['name']
            update_status_row(name, False, objects_in_hands)  # ✅ Обновляем статус как "Не на месте"
            del tracked_people[tid]
            if tid in pending_faces:
                del pending_faces[tid]
            continue


        if tid in pending_faces and tid not in tracked_people:
            continue  # ещё распознается, пока не переходит в tracked_people

        if tid not in tracked_people:
            continue  # ни в pending_faces, ни в tracked_people


        name = tracked_people[tid]['name']
        x1, y1, x2, y2 = tracked_people[tid]['bbox']
        in_zone = bbox_intersects_polygon((x1, y1, x2, y2), zone_bbox)
        name = tracked_people[tid]['name']
        now = time.time()


        # Обновляем переменную name из актуального состояния tracked_people
        name = tracked_people[tid]['name']

        update_status_row(name, in_zone, objects_in_hands)

        # Обновляем статус в Tkinter
        # if tracked_people[tid].get("last_objects", []) != objects_in_hands:
        #     update_status_with_objects(name, objects_in_hands)
        #     tracked_people[tid]["last_objects"] = objects_in_hands


        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id} - {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=3)

    frame_resized = cv2.resize(frame, output_resolution)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face + Body Tracking", frame_resized)

    # Проверка, закрыто ли окно Tkinter
    if not root.winfo_exists():
        print("[INFO] Окно Tkinter закрыто. Завершение работы.")
        break
    root.update()

    # Проверка закрытия окна OpenCV
    if cv2.getWindowProperty("Face + Body Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
root.destroy()
