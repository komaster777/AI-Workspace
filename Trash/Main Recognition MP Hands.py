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
# Инициализация модели отслеживания рук с использованием GPU
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)


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
recognition_t = 0.6
required_size = (160, 160)

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


# Функция для вычисления пересечения двух прямоугольников
def rects_intersect(body_rect, obj_rect):
    x1_body, y1_body, x2_body, y2_body = body_rect
    x1_obj, y1_obj, x2_obj, y2_obj = obj_rect

    # Вычисляем координаты пересечения
    x_left = max(x1_body, x1_obj)
    y_top = max(y1_body, y1_obj)
    x_right = min(x2_body, x2_obj)
    y_bottom = min(y2_body, y2_obj)

    # Проверяем, есть ли пересечение
    if x_left < x_right and y_top < y_bottom:
        return True  # Прямоугольники пересекаются
    return False


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
    hand_results = hands.process(frame_rgb)

    # Обнаруживаем объекты с помощью YOLO
    object_results = object_model.predict(
        frame,
        imgsz=640,  # Размер входного изображения
        conf=0.6,  # Порог уверенности, чем ниже, чтобы модель распознавала больше объектов
        iou=0.4,  # Порог пересечения IoU, чем ниже, тем учитывать частичное перекрытие объектов.
        verbose=False
    )[0]

    # Список классов, которые нужно игнорировать
    ignored_classes = ["person", "car", "cat"]

    if object_results.boxes is not None and len(object_results.boxes) > 0:
        for box, conf, cls in zip(object_results.boxes.xyxy, object_results.boxes.conf,
                                  object_results.boxes.cls):
            class_name = object_model.names[int(cls)]
            if class_name not in ignored_classes:
                x1, y1, x2, y2 = map(int, box.tolist())
                # Рисуем прямоугольник вокруг объекта
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{object_model.names[int(cls)]} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #  Руки MediaPipe
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Координаты для прямоугольника руки
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Обходим все ключевые точки руки
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Рисуем прямоугольник вокруг руки
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Рисуем линии между точками руки
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                      mp_hands.HAND_CONNECTIONS)


    objects_in_hands = []  # Список объектов, находящихся в руках


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

                # Код для детекции предметов в руках
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Координаты для прямоугольника руки
                        h, w, _ = frame.shape
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0

                        if object_results.boxes is not None and len(object_results.boxes) > 0:
                            # Проверяем пересечение объектов с прямоугольником руки
                            for box, conf, cls in zip(object_results.boxes.xyxy, object_results.boxes.conf,
                                                      object_results.boxes.cls):
                                x1, y1, x2, y2 = map(int, box.tolist())
                                object_rect = (x1, y1, x2, y2)
                                hand_rect = (x_min, y_min, x_max, y_max)

                                # Если прямоугольники пересекаются
                                if rects_intersect(hand_rect, object_rect):
                                    objects_in_hands.append({
                                        "object": object_model.names[int(cls)],
                                        "confidence": float(conf.item()),
                                        "bbox": (x1, y1, x2, y2)
                                    })






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

        # # Код для детекции предметов в руках через прямоугольник тела
        # if object_results.boxes is not None and len(object_results.boxes) > 0:
        #     # Проверяем пересечение объектов с прямоугольником руки
        #     for box, conf, cls in zip(object_results.boxes.xyxy, object_results.boxes.conf,
        #                               object_results.boxes.cls):
        #         class_name = object_model.names[int(cls)]
        #         if class_name not in ignored_classes:
        #             x1, y1, x2, y2 = map(int, box.tolist())
        #             object_rect = (x1, y1, x2, y2)
        #             x_min, y_min, x_max, y_max = tracked_people[tid]['bbox']
        #             body_rect = (x_min, y_min, x_max, y_max)
        #             x1, y1, x2, y2 = tracked_people[tid]['bbox']
        #
        #             # Если прямоугольники пересекаются
        #             if rects_intersect(body_rect, object_rect):
        #                 objects_in_hands.append({
        #                     "object": object_model.names[int(cls)],
        #                     "confidence": float(conf.item()),
        #                     "bbox": (x1, y1, x2, y2)
        #                 })


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
