import os
import cv2
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk  # Для создания ползунка
import re  # Для проверки имени

DATASET_PATH = "dataset/"

# Функция для съемки и сохранения лица
def capture_face():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось подключиться к камере.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Ошибка", "Не удалось получить изображение с камеры.")
            break

        # Отображаем видео
        cv2.putText(frame, "Press 's' to save, 'q' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Захват лица", frame)

        key = cv2.waitKey(1) & 0xFF

        # Нажать 's' для сохранения фото, 'q' для отмены
        if key == ord('s'):
            # Окно для ввода имени и выбора существующей папки
            def select_name():
                while True:
                    name = simpledialog.askstring("Имя", "Введите имя для нового лица:")

                    if not name:
                        return None

                    # Проверка, чтобы имя состояло только из латинских символов
                    if not re.match("^[A-Za-z]+$", name):
                        messagebox.showerror("Ошибка", "Имя должно содержать только латинские символы.")
                    else:
                        person_folder = os.path.join(DATASET_PATH, name)
                        # Проверка на существование папки
                        if not os.path.exists(person_folder):
                            create_folder = messagebox.askyesno("Папка не существует", f"Папки {name} не существует. Создать?")
                            if create_folder:
                                os.makedirs(person_folder)
                            else:
                                return None
                        return person_folder

            person_folder = select_name()
            if person_folder is None:
                break

            # Снимаем фото и сохраняем
            # Вычисление следующего номера для изображения
            image_count = len([f for f in os.listdir(person_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]) + 1
            image_path = os.path.join(person_folder, f"{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            messagebox.showinfo("Успех", f"Изображение сохранено: {image_path}")

            break

        elif key == ord('q'):
            messagebox.showinfo("Отмена", "Съемка отменена.")
            break

    cap.release()
    cv2.destroyAllWindows()


# Функция для обновления базы
def update_database():
    subprocess.run(["python", "Train Face.py"], check=True)
    messagebox.showinfo("Готово", "База эмбеддингов обновлена!")


# Функция для запуска Face Recognition
def start_recognition():
    scale_value = scale_var.get()  # Получаем текущий масштаб
    width = width_var.get()  # Получаем ширину
    height = height_var.get()  # Получаем высоту
    subprocess.Popen(["python", "Main Recognition.py", str(scale_value), str(width), str(height)])


# Функция для обновления метки текущего значения масштаба
def update_scale_label(value):
    scale_value_label.config(text=f"{float(value):.2f}")


# Создание окна
root = tk.Tk()
root.title("Настройки Face & Body Recognition")

# Устанавливаем размер окна
root.geometry("400x400")

btn_capture = tk.Button(root, text="📸 Снять и сохранить лицо", command=capture_face)
btn_capture.pack(pady=10)

btn_update = tk.Button(root, text="🔄 Обновить базу", command=update_database)
btn_update.pack(pady=10)

# Ползунок для регулирования масштаба
scale_var = tk.DoubleVar(value=1.0)  # Значение по умолчанию
scale_label = tk.Label(root, text="Масштаб окна (например, 1.0):")
scale_label.pack(pady=5)

scale_frame = tk.Frame(root)
scale_frame.pack(pady=5)

scale_slider = ttk.Scale(scale_frame, from_=0.5, to=2.0, orient="horizontal", variable=scale_var, command=update_scale_label)
scale_slider.pack(side="left", padx=5)

# Метка для отображения текущего значения масштаба
scale_value_label = tk.Label(scale_frame, text=f"{scale_var.get():.2f}")
scale_value_label.pack(side="left")

# Поля для ввода ширины и высоты
resolution_label = tk.Label(root, text="Разрешение видео (например, 1920 x 1080):")
resolution_label.pack(pady=5)

resolution_frame = tk.Frame(root)
resolution_frame.pack(pady=5)

width_var = tk.IntVar(value=1920)  # Значение ширины по умолчанию
height_var = tk.IntVar(value=1080)  # Значение высоты по умолчанию

width_entry = tk.Entry(resolution_frame, textvariable=width_var, width=10)
width_entry.pack(side="left", padx=5)

x_label = tk.Label(resolution_frame, text="x")
x_label.pack(side="left")

height_entry = tk.Entry(resolution_frame, textvariable=height_var, width=10)
height_entry.pack(side="left", padx=5)

btn_start = tk.Button(root, text="▶ Запустить Face & Body Recognition", command=start_recognition)
btn_start.pack(pady=10)

root.mainloop()