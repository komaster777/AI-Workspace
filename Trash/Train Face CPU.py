import os
import cv2
import numpy as np
import pickle
import torch
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ✅ Используем GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используем устройство: {device}")

DATASET_PATH = "dataset/"
OUTPUT_FILE = "face_encodings.pkl"
REQUIRED_SIZE = (160, 160)

# ✅ Инициализация моделей
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ✅ Преобразования для PyTorch
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(REQUIRED_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для FaceNet
])

# ✅ Удаление старой базы перед обновлением
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"[INFO] Старая база {OUTPUT_FILE} удалена.")

# ✅ Инициализация новых списков для эмбеддингов и имен
known_encodings, known_names = [], []

def process_image(image_path):
    """Создает эмбеддинг лица из изображения"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            face = image_rgb[y:y + height, x:x + width]

            if face.shape[0] > 0 and face.shape[1] > 0:
                face_tensor = transform(face).unsqueeze(0).to(device)  # Преобразуем в PyTorch тензор
                with torch.no_grad():
                    encoding = facenet(face_tensor).cpu().numpy().flatten()
                return encoding
    return None

# ✅ Проход по папкам и добавление новых лиц
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Пропускаем только файлы с лицами
                encoding = process_image(image_path)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                    print(f"[INFO] Добавлено: {person_name} ({image_name})")

# ✅ Сохранение обновленной базы
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": np.array(known_encodings), "names": known_names}, f)

print(f"[✅] Файл эмбеддингов обновлен: {OUTPUT_FILE}")
