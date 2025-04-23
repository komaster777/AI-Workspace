import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- Гиперпараметры ---
EPOCHS = 10
BATCH_SIZE = 32  # Попробуйте 32, 64 или 128 в зависимости от вашей задачи и ресурсов
LEARNING_RATE = 0.001
INPUT_SIZE = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 2  # "Читает" или "Не читает"

# --- Загрузка данных ---
df = pd.read_csv("pose_data.csv")
X = df.iloc[:, :-1].values  # Все столбцы, кроме последнего (метка)
y = df.iloc[:, -1].values   # Последний столбец (метка)

# Преобразуем в тензоры
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# --- Датасет ---
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.view(X.shape[0], 1, -1)  # Добавляем 1D размерности для LSTM
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Модель LSTM ---
class PoseLSTM(nn.Module):
    def __init__(self):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Используем последний выход LSTM
        return out

# --- Инициализация модели ---
model = PoseLSTM()

# Проверка наличия сохраненной модели
model_path = 'best_model.pth'
if os.path.exists(model_path):
    print(f"Загружаем модель из {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Модель не найдена, начнем обучение с нуля.")

# --- Обучение ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Создаем DataLoader
dataset = PoseDataset(X, y)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

best_loss = float('inf')

# Проверка доступности GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU доступен:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU не доступен, используется CPU")

# Перемещение модели на GPU
model.to(device)

patience = 5  # Количество эпох, после которых останавливаем обучение
counter = 0
best_loss = float('inf')

# Обучение
for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Перемещение данных на GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Эпоха [{epoch + 1}/{EPOCHS}], Средняя потеря: {average_loss:.4f}")

    # Проверка на улучшение
    if average_loss < best_loss:
        best_loss = average_loss
        counter = 0  # Сброс счетчика
        # Сохранение модели
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'best_model.pth')
        print("Модель обновлена и сохранена в best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Обучение остановлено из-за отсутствия улучшений.")
            break

print("Обучение завершено.")
