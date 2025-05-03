import os
import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from ultralytics import YOLO

# ✅ Используем GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используем устройство: {device}")

DATASET_PATH = "dataset/"
OUTPUT_FILE = "face_encodings.pkl"
REQUIRED_SIZE = (160, 160)

# ✅ Инициализация моделей
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_model = YOLO("yolov8n-face.pt").to(device)
face_model.to("cuda")  # Явно переносим на GPU

# Проверка, работает ли YOLO face на GPU
face_model_device = face_model.device
if face_model_device.type == 'cuda':
    print(f"[INFO] YOLO face модель работает на GPU: {face_model_device}")
else:
    print(f"[INFO] YOLO face модель работает на CPU")

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
    # Используем YOLO для детекции лиц
    results = face_model.predict(image_rgb, imgsz=640, conf=0.5, verbose=False)[0]

    if results.boxes is not None:  # Проверяем, найдены ли лица
        for box in results.boxes.xyxy:  # Координаты боксов
            x1, y1, x2, y2 = map(int, box.tolist())  # Извлекаем координаты
            face = image_rgb[y1:y2, x1:x2]  # Вырезаем область лица

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

# ✅ Сохранение эмбеддингов как тензоров
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"[✅] Файл эмбеддингов обновлен: {OUTPUT_FILE}")
print(f"[INFO] Всего лиц: {len(known_names)}")
# Указываем путь к директории
directory_path = "dataset/"
folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
print("Имена:", folder_names)

