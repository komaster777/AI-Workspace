import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import subprocess
import tkinter as tk
import json
from PIL import Image, ImageTk
from tkinter import simpledialog, messagebox
from tkinter import ttk  # Для создания ползунка
import re  # Для проверки имени

DATASET_PATH = "dataset/"


# Функция для съемки и сохранения лица
def capture_face():
    scale_value = scale_var.get()  # Получаем текущий масштаб
    width, height = resolution_options[selected_resolution.get()]  # Получаем высоту и ширину окна
    output_resolution = (int(width * scale_value), int(height * scale_value))

    width_cam, height_cam = resol_cam_options[selected_resol_cam.get()] # Получаем разрешенеи камеры


    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось подключиться к камере.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Ошибка", "Не удалось получить изображение с камеры.")
            break

        frame_resized = cv2.resize(frame, output_resolution)
        # Копируем оригинальный кадр, чтобы сохранить его без текста
        original_frame = frame.copy()

        # Отображаем видео
        cv2.putText(frame, "Press 's' to save, 'q','esc' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face capture", frame)

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
            cv2.imwrite(image_path, original_frame)
            messagebox.showinfo("Успех", f"Изображение сохранено: {image_path}")

            break

        elif key == ord('q') or key == 27:
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
    width, height = resolution_options[selected_resolution.get()] # Получаем высоту и ширину окна
    width_cam, height_cam = resol_cam_options[selected_resol_cam.get()]  # Получаем высоту и ширину окна
    rec_t_value = sel_rec_t.get()
    hands_r_value = sel_hands_r.get()
    vector_r_value = sel_vector_r.get()
    subprocess.Popen(["python", "Main Recognition.py", str(scale_value), str(width), str(height), str(width_cam), str(height_cam),
                      str(rec_t_value), str(hands_r_value), str(vector_r_value)])


# Функция для обновления метки текущего значения масштаба
def update_scale_label(value):
    scale_value_label.config(text=f"{float(value):.2f}")

# Функция для определения рабочей зоны
def define_work_zone():
    scale_value = scale_var.get()  # Получаем текущий масштаб
    width, height = resolution_options[selected_resolution.get()]  # Получаем высоту и ширину окна
    output_resolution = (int(width * scale_value), int(height * scale_value))
    width_cam, height_cam = resol_cam_options[selected_resol_cam.get()]  # Получаем разрешенеи камеры

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    # Устанавливаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    cap.release()


    if not ret:
        messagebox.showerror("Ошибка", "Не удалось получить изображение с камеры.")
        return

    frame_resized = cv2.resize(frame, output_resolution)

    # Сохраняем снимок
    snapshot_path = "workzone_snapshot.jpg"
    cv2.imwrite(snapshot_path, frame_resized)



    # Загружаем изображение для Tkinter
    img = Image.open(snapshot_path)
    tk_img = ImageTk.PhotoImage(img)

    # Создаем новое окно для редактирования зоны
    zone_window = tk.Toplevel(root)
    zone_window.title("Обозначьте рабочую зону")

    canvas = tk.Canvas(zone_window, width=img.width, height=img.height)
    canvas.pack()

    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img  # нужно сохранить ссылку, чтобы не удалялось

    # Начальные 4 точки (прямоугольник в центре)
    h, w = img.height, img.width
    points = [
        [w // 3, h // 3],
        [2 * w // 3, h // 3],
        [2 * w // 3, 2 * h // 3],
        [w // 3, 2 * h // 3]
    ]
    point_circles = []
    polygon = None

    def draw_zone():
        nonlocal polygon
        if polygon:
            canvas.delete(polygon)
        polygon = canvas.create_polygon(*sum(points, []), outline="red", width=2, fill="", tags="zone")

        for c in point_circles:
            canvas.delete(c)
        point_circles.clear()

        for i, (x, y) in enumerate(points):
            circle = canvas.create_oval(x-5, y-5, x+5, y+5, fill="blue", tags=f"pt{i}")
            point_circles.append(circle)

    selected_point = [None]  # Используем список, чтобы менять из вложенной функции

    def on_press(event):
        item = canvas.find_withtag("current")
        tags = canvas.gettags(item)
        for tag in tags:
            if tag.startswith("pt"):
                selected_point[0] = int(tag[2:])  # индекс активной точки
                break

    def on_release(event):
        selected_point[0] = None  # Отпустили — сбросили выбор

    def on_drag(event):
        idx = selected_point[0]
        if idx is not None:
            x = max(0, min(event.x, w))
            y = max(0, min(event.y, h))
            points[idx] = [x, y]
            draw_zone()

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<B1-Motion>", on_drag)

    draw_zone()

    # Кнопка сохранения координат
    def save_zone():
        # Сохраняем масштаб и разрешение в JSON
        zone_data = {
            "points": points,
            "scale": scale_var.get(),  # сохраняем выбранный масштаб
            "resolution": resolution_options[selected_resolution.get()],
            "resolution_cam": resol_cam_options[selected_resol_cam.get()]
        }
        with open("work_zone.json", "w") as f:
            json.dump(zone_data, f)
        messagebox.showinfo("Готово", "Рабочая зона сохранена в work_zone.json")
        zone_window.destroy()

    btn_save = tk.Button(zone_window, text="💾 Сохранить зону", command=save_zone)
    btn_save.pack(pady=10)

    zone_window.bind("<Return>", lambda event: save_zone())  # ← Привязка Enter

# Создание окна
root = tk.Tk()
root.title("Настройки Face & Body Recognition")

# Устанавливаем размер окна
root.geometry("400x600")

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

# Поля для ввода ширины и высоты для окна
resolution_label = tk.Label(root, text="Разрешение окна:")
resolution_label.pack(pady=5)
resolution_options = {
    "2560 x 1440": (2560, 1440),
    "1920 x 1080": (1920, 1080),
    "1600 x 900": (1600, 900),
    "1280 x 720": (1280, 720),
    "800 x 600": (800, 600),
    "640 x 480": (640, 480)
}
selected_resolution = tk.StringVar(value="1920 x 1080")

resolution_combobox = ttk.Combobox(root, textvariable=selected_resolution, values=list(resolution_options.keys()), state="readonly")
resolution_combobox.pack(pady=5)

# Поля для ввода ширины и высоты для камеры
resol_cam_label = tk.Label(root, text="Разрешение камеры:")
resol_cam_label.pack(pady=5)
resol_cam_options = {
    "2560 x 1440": (2560, 1440),
    "1920 x 1080": (1920, 1080),
    "1600 x 900": (1600, 900),
    "1280 x 720": (1280, 720),
    "800 x 600": (800, 600),
    "640 x 480": (640, 480)
}
selected_resol_cam = tk.StringVar(value="2560 x 1440")

resol_cam_combobox = ttk.Combobox(root, textvariable=selected_resol_cam, values=list(resol_cam_options.keys()), state="readonly")
resol_cam_combobox.pack(pady=5)

btn_start = tk.Button(root, text="▶ Запустить Face & Body Recognition", command=start_recognition)
btn_start.pack(pady=10)

btn_zone = tk.Button(root, text="◼ Обозначить рабочую зону", command=define_work_zone)
btn_zone.pack(pady=(10, 0))
note1 = tk.Label(root, text="Примечание: лучше сохранять зону с теми параметрами камеры, ")
note1.pack(pady=(0, 0))
note2 = tk.Label(root, text="с которыми будет запускаться распознавание (соотношение сторон)")
note2.pack(pady=(0, 5))

Dop = tk.Label(root, text="Дополнительные параметры")
Dop.pack(pady=(0, 5))
Dop1 = tk.Label(root, text="Порог распознания:")
Dop1.pack(pady=(0, 1))

sel_rec_t = tk.StringVar(value="0.6")
rec_t = tk.Entry(root, textvariable=sel_rec_t, width=6)
rec_t.pack(pady=5)

Dop2 = tk.Label(root, text="Радиус рук (порог распознания: от центра объекта до центра руки):")
Dop2.pack(pady=(0, 1))

sel_hands_r = tk.StringVar(value="300")
hands_r = tk.Entry(root, textvariable=sel_hands_r, width=8)
hands_r.pack(pady=5)

Dop3 = tk.Label(root, text="Длина, на которую нужно сместить точку руки от локтя):")
Dop3.pack(pady=(0, 1))

sel_vector_r = tk.StringVar(value="150")
vector_r = tk.Entry(root, textvariable=sel_vector_r, width=8)
vector_r.pack(pady=5)

# Завершение по ESC
def on_escape(event=None):

    cv2.destroyAllWindows()
    root.quit()

root.bind("<Escape>", on_escape)

root.mainloop()

