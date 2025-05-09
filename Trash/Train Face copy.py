import os
import cv2
import numpy as np
import pickle
import torch
from insightface.app import FaceAnalysis
from torchvision import transforms
from ultralytics import YOLO

# ✅ Используем GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Используем устройство: {device}")

DATASET_PATH = "dataset/"
OUTPUT_FILE = "face_encodings.pkl"
REQUIRED_SIZE = (160, 160)

# ✅ Инициализация моделей
# ✅ Инициализация InsightFace для детекции и извлечения эмбеддингов
face_app = FaceAnalysis(name="buffalo_l")  # Используем предобученный пакет
face_app.prepare(ctx_id=0 if device == "cuda" else -1)



# # ✅ Преобразования для PyTorch
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(REQUIRED_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для FaceNet
# ])

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

    # Используем InsightFace для детекции лиц
    faces = face_app.get(image_rgb)
    if faces:  # Если лица найдены
        for face in faces:
            bbox = face.bbox.astype(int)  # Координаты лица
            embedding = face.embedding  # Эмбеддинг лица
            x1, y1, x2, y2 = bbox
            print(f"[INFO] Лицо найдено: координаты ({x1}, {y1}, {x2}, {y2})")
            return embedding
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
    pickle.dump({"encodings": np.array(known_encodings), "names": known_names}, f)

print(f"[✅] Файл эмбеддингов обновлен: {OUTPUT_FILE}")
print(f"[INFO] Всего лиц: {len(known_names)}")
# Указываем путь к директории
directory_path = "dataset/"
folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
print("Имена:", folder_names)

