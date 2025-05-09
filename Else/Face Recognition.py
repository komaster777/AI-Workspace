import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist

# ✅ Автоматическое включение GPU (если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используем устройство: {device}")

# ✅ Инициализация детектора лиц (Mediapipe)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ✅ Инициализация FaceNet (InceptionResnetV1) для создания эмбеддингов
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
required_size = (160, 160)
recognition_t = 0.75  # Порог идентификации

# ✅ Разрешение вывода видео (можно изменить)
output_resolution = (1920, 1080)  # Например, (1920, 1080) или (640, 480)

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

# ✅ Настройка камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # Исходное разрешение
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)  # Исходное разрешение
cap.set(cv2.CAP_PROP_FPS, 30)

# ✅ Регулирование FPS
desired_fps = 30
frame_time = 1.0 / desired_fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Не удалось получить кадр с камеры.")
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        h_img, w_img, _ = frame.shape
        x, y, w, h = max(0, x), max(0, y), min(w, w_img - x), min(h, h_img - y)

        face_image = frame[y:y + h, x:x + w]

        if face_image is None or face_image.size == 0:
            print("[ERROR] Лицо не найдено или вышло за границы кадра.")
            continue

        # ✅ Обработка изображения и получение эмбеддингов через PyTorch
        face_image_resized = cv2.resize(face_image, required_size)
        face_image_tensor = torch.tensor(face_image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        face_image_tensor = face_image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            face_encoding = facenet(face_image_tensor).cpu().numpy()

        # ✅ Проверяем форму массивов перед вычислением расстояний
        known_face_encodings = np.array(known_face_encodings)
        face_encoding = face_encoding.reshape(1, -1)

        # ✅ Вычисление расстояний
        distances = cdist(known_face_encodings, face_encoding, metric="cosine").flatten()
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

    # ✅ Изменение разрешения вывода
    frame_resized = cv2.resize(frame, output_resolution)

    # ✅ Отображение FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame_resized)

    # ✅ Ограничение FPS (если нужно)
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_time - elapsed_time)
    time.sleep(sleep_time)

    if cv2.waitKey(5) == 27:  # Нажатие ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()
