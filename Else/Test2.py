import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import os
import time

# Укажите путь к файлу модели
model_path = 'face_landmarker.task'

# Вспомогательные функции для визуализации результатов
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    return annotated_image

def is_eye_closed(landmarks, left_eye_indices, right_eye_indices):
    left_eye_aspect_ratio = calculate_eye_aspect_ratio(landmarks, left_eye_indices)
    right_eye_aspect_ratio = calculate_eye_aspect_ratio(landmarks, right_eye_indices)
    if left_eye_aspect_ratio < 0.2 and right_eye_aspect_ratio < 0.2:
        return "Closed"
    else:
        return "Open"

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    eye = np.array([(point.x, point.y) for point in eye])
    horizontal_dist = np.linalg.norm(eye[0] - eye[3])
    vertical_dist1 = np.linalg.norm(eye[1] - eye[5])
    vertical_dist2 = np.linalg.norm(eye[2] - eye[4])
    return (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

# Инициализация FaceLandmarker
base_options = BaseOptions(model_asset_path=model_path)
options = FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Открытие видеопотока с веб-камеры с измененными параметрами
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # Установить ширину кадра
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)  # Установить высоту кадра
cap.set(cv2.CAP_PROP_FPS, 30)  # Установить количество кадров в секунду

# Изменение размера выходного окна
window_name = 'Annotated Image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

# Индексы точек лицевого меша для глаз
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить изображение с веб-камеры")
        break

    # Преобразование изображения в RGB формат
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка изображения
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(image)

    # Визуализация результатов
    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

    # Расчет FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"

    # Распознавание состояния глаз
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        eye_state = is_eye_closed(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
    else:
        eye_state = "Unknown"

    # Отображение текста FPS и состояния глаз
    cv2.putText(annotated_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_image, f"Eyes: {eye_state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите Esc для выхода
        break

cap.release()
cv2.destroyAllWindows()