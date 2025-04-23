from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
import cv2
from django.apps import apps
from .models import *
from django.shortcuts import render, get_object_or_404
import numpy as np
import mediapipe as mp
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import random
from PIL import Image, ImageDraw, ImageFont

def media_file_filter_opencv(request, id):
    # Функция, генерирующая поток данных изображения.
    def generate(media_file_id, filter_type='RGB'):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        media_file_path = media_file.video_file.path
        cap = cv2.VideoCapture(media_file_path)
        while True:
            ret, img = cap.read()
            # Применение выбранного фильтра
            if filter_type != 'RGB':
                img = cv2.cvtColor(img, getattr(cv2, 'COLOR_BGR2' + filter_type))
            # ... добавьте условия для других фильтров ...
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
            cv2.waitKey(24)
    # Возвращаем потоковый HTTP-ответ с потоком изображения.
    filter_type = request.GET.get('filter', 'RGB')
    return StreamingHttpResponse(generate(id, filter_type), content_type='multipart/x-mixed-replace; boundary=frame')

def media_file_mask_opencv(request, id):
    # Получение параметров фильтра из запроса
    h1 = int(request.GET.get('h1', 0))
    s1 = int(request.GET.get('s1', 0))
    v1 = int(request.GET.get('v1', 0))
    h2 = int(request.GET.get('h2', 255))
    s2 = int(request.GET.get('s2', 255))
    v2 = int(request.GET.get('v2', 255))

    def stream_video_with_filter(media_file_id, h1, s1, v1, h2, s2, v2):
        video = get_object_or_404(Media_file, id=media_file_id)
        video_path = video.video_file.path
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            # Конвертация изображения в HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Создание массивов numpy для границ фильтра
            lower_bound = np.array([h1, s1, v1], dtype="uint8")
            upper_bound = np.array([h2, s2, v2], dtype="uint8")
            # Применение цветового фильтра
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Кодирование кадра для передачи через HTTP
            _, encoded_frame = cv2.imencode('.jpg', result)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
            cv2.waitKey(24)
    return StreamingHttpResponse(stream_video_with_filter(id, h1, s1, v1, h2, s2, v2),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

def media_file_contours_opencv(request, id):
    # Получение параметров фильтра из запроса
    h1 = int(request.GET.get('h1', 0))
    s1 = int(request.GET.get('s1', 0))
    v1 = int(request.GET.get('v1', 0))
    h2 = int(request.GET.get('h2', 255))
    s2 = int(request.GET.get('s2', 255))
    v2 = int(request.GET.get('v2', 255))

    def stream_video_with_contours(media_file_id, h1, s1, v1, h2, s2, v2):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        media_file_path = media_file.video_file.path
        cap = cv2.VideoCapture(media_file_path)
        font_path = "/Users/mak/PycharmProjects/pythonProject3/mysite/21028.ttf"  # Укажите путь к шрифту
        font = ImageFont.truetype(font_path, 24)  # Размер шрифта

        while True:
            ret, frame = cap.read()
            # Конвертация изображения в HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Создание массивов numpy для границ фильтра
            lower_bound = np.array([h1, s1, v1], dtype="uint8")
            upper_bound = np.array([h2, s2, v2], dtype="uint8")
            # Нахождение и отрисовка контуров
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    # Создаем PIL-изображение из фрейма
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    # Генерируем случайное число
                    random_number = random.randint(80, 90)
                    # Добавляем текст на PIL-изображение
                    draw.text((x, y - 30), f"Шоколад 0.{random_number}", font=font, fill=(0, 255, 0))
                    # Конвертируем обратно в OpenCV
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # Кодирование кадра для передачи через HTTP
            _, encoded_frame = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
            cv2.waitKey(50)
    return StreamingHttpResponse(stream_video_with_contours(id, h1, s1, v1, h2, s2, v2),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

def media_file_pose_opencv(request, id):
    def stream_pose(media_file_id):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        media_file_path = media_file.video_file.path
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        cap = cv2.VideoCapture(media_file_path)
        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # print(results.pose_landmarks)
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            _, encoded_frame = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
    return StreamingHttpResponse(stream_pose(id), content_type='multipart/x-mixed-replace; boundary=frame')

def media_file_YOLO_opencv(request, id):
    def stream_YOLO(media_file_id):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        Photo_path = Media_file.image.path
        cap = cv2.VideoCapture(Photo_path)
        model = YOLO('best.pt')  # load a pretrained model (recommended for training)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Предсказание модели для каждого кадра
            results = model.predict(frame)
            # Получение кадра с наложенными результатами
            frame_with_results = results[0].plot()
            # Показать кадр
            ret, jpeg = cv2.imencode('.jpg', frame_with_results)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # Destroy all the windows
    return StreamingHttpResponse(stream_YOLO(id), content_type='multipart/x-mixed-replace; boundary=frame')

