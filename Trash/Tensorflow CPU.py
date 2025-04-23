import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import tensorflow as tf
from keras_facenet import FaceNet
from scipy.spatial.distance import cdist

# ✅ Автоматическое включение GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("[INFO] GPU включен")
else:
    print("[INFO] GPU не найден, используем CPU")

# ✅ Инициализация детектора лиц (Mediapipe)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ✅ Инициализация FaceNet для создания эмбеддингов
facenet = FaceNet()
required_size = (160, 160)
recognition_t = 0.5  # Порог идентификации


def detect_faces(image):
    """Использует Mediapipe для быстрого обнаружения лиц"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            faces.append((x, y, width, height))
    return faces


def load_encodings(path="face_encodings.pkl"):
    """Загружает сохраненные эмбеддинги лиц из файла"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["encodings"]), data["names"]


# ✅ Загрузка обученной базы эмбеддингов
known_face_encodings, known_face_names = load_encodings()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

# Установим желаемое количество FPS
desired_fps = 60  # Желаемый FPS
frame_time = 1.0 / desired_fps  # Время для каждого кадра

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Не удалось получить кадр с камеры.")
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        h_img, w_img, _ = frame.shape

        # Ограничение координат, чтобы избежать выхода за границы
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        face_image = frame[y:y + h, x:x + w]

        if face_image is None or face_image.size == 0:
            print("[ERROR] Лицо не найдено или вышло за границы кадра.")
            continue

        face_image_resized = cv2.resize(face_image, required_size)
        face_encoding = facenet.embeddings([face_image_resized])[0]

        # ✅ Векторное вычисление расстояний (вместо for-цикла)
        distances = cdist(known_face_encodings, [face_encoding], metric="cosine").flatten()
        min_distance = np.min(distances)
        best_match = np.argmin(distances)

        # ✅ Проверяем, есть ли совпадение
        name = "Unknown"
        if min_distance < recognition_t:
            name = known_face_names[best_match]

        # ✅ Рисуем рамку и имя
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({(1 - min_distance) * 100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2)

    # ✅ Отображение FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(5) == 27:
        break

cap.release()
cv2.destroyAllWindows()