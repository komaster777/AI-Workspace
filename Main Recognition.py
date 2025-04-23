import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import torch
import sys  # Для получения аргументов командной строки
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist

# ✅ Автоматическое включение GPU (если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используем устройство: {device}")

# ✅ Инициализация детектора лиц (Mediapipe)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ✅ Инициализация Mediapipe Pose для отслеживания тела
mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ Инициализация FaceNet (InceptionResnetV1) для создания эмбеддингов
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
required_size = (160, 160)
recognition_t = 0.75  # Порог идентификации

# ✅ Получение масштаба и разрешения из аргументов командной строки
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
height = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
print(f"[INFO] Масштаб окна: {scale}")
print(f"[INFO] Разрешение видео: {width} x {height}")

# ✅ Разрешение вывода видео
output_resolution = (int(width * scale), int(height * scale))

# ✅ Словарь для привязки тела к распознанному лицу
tracked_people = {}

# ✅ Функция для обнаружения лиц
def detect_faces(image):
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

# ✅ Функция загрузки эмбеддингов
def load_encodings(path="face_encodings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["encodings"]), data["names"]

# ✅ Загрузка обученной базы
known_face_encodings, known_face_names = load_encodings()

# ✅ Настройка камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_tracker.process(frame_rgb)

    body_x_min, body_y_min = float('inf'), float('inf')
    body_x_max, body_y_max = 0, 0

    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]

        if face_image is None or face_image.size == 0:
            print("[ERROR] Лицо не найдено или вышло за границы кадра.")
            continue

        face_image_resized = cv2.resize(face_image, required_size)
        face_image_tensor = torch.tensor(face_image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        face_image_tensor = face_image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            face_encoding = facenet(face_image_tensor).cpu().numpy()

        distances = cdist(known_face_encodings, face_encoding, metric="cosine").flatten()
        min_distance = np.min(distances)
        best_match = np.argmin(distances)

        name = "Unknown"
        if min_distance < recognition_t:
            name = known_face_names[best_match]
            tracked_people[name] = time.time()

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({(1 - min_distance) * 100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if pose_results.pose_landmarks:
        h, w, _ = frame.shape
        for landmark in pose_results.pose_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            body_x_min, body_y_min = min(body_x_min, x), min(body_y_min, y)
            body_x_max, body_y_max = max(body_x_max, x), max(body_y_max, y)

        for name in list(tracked_people.keys()):
            last_seen = tracked_people[name]
            if time.time() - last_seen < 3:
                # Рисуем прямоугольник вокруг тела
                cv2.rectangle(frame, (body_x_min, body_y_min), (body_x_max, body_y_max), (255, 0, 0), 2)
                cv2.putText(frame, name, (body_x_min, body_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                del tracked_people[name]

    frame_resized = cv2.resize(frame, output_resolution)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face & Body Recognition", frame_resized)

    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_time - elapsed_time)
    time.sleep(sleep_time)

    if cv2.waitKey(5) == 27:
        break

cap.release()
cv2.destroyAllWindows()