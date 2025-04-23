import cv2
import os
import face_recognition
import numpy as np


# if __name__ == '__main__':
#     def nothing(*arg):
#         pass

# Путь к папке с данными
dataset_path = "../dataset/"

# Список для хранения кодировок лиц и имен
known_face_encodings = []
known_face_names = []

# Проходим по всем папкам в директории dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Проверяем, является ли это папкой
    if os.path.isdir(person_folder):
        # Проходим по всем изображениям в папке
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Проверяем, является ли файл изображением
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Загружаем изображение
                image = face_recognition.load_image_file(image_path)

                # Получаем кодировки лиц (на изображении может быть несколько лиц)
                face_encodings = face_recognition.face_encodings(image)

                # Если на изображении есть лица, добавляем их в список
                for face_encoding in face_encodings:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)  # создаем главное окно
cv2.resizeWindow("result", 1280, 720)  # устанавливаем размер окна

cap = cv2.VideoCapture(0)

# Устанавливаем разрешение камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # ширина
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # высота

# Устанавливаем частоту кадров
cap.set(cv2.CAP_PROP_FPS, 30)  # например, 30 FPS

# cv2.namedWindow("settings")  # создаем окно настроек



# cv2.createTrackbar('Brightness', 'settings', 10, 100, nothing)
# cv2.createTrackbar('Contrast', 'settings', 1, 100, nothing)

frame_skip = 3  # Пропустите 3 кадра
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Проверка на успешное чтение кадра

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Пропускаем обработку этого кадра

    # # считываем значение яркости
    # brightness = cv2.getTrackbarPos('Brightness', 'settings')
    # contrast = cv2.getTrackbarPos('Contrast', 'settings')
    #
    # # Изменяем яркость
    # result = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Преобразуем изображение в формат RGB
    #rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ищем все лица на текущем кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Сравниваем каждое лицо с известными лицами
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Сравниваем полученное лицо с известными
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Если лицо совпадает с известным
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Пишем имя (если распознано)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)


    # Показываем результат
    cv2.imshow("result", frame)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()

