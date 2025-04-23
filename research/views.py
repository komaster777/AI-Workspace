from django.shortcuts import render, get_object_or_404, redirect
from .forms import *
from django.core.paginator import Paginator
from django.http import StreamingHttpResponse, JsonResponse
import cv2
from .models import *
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import random
from PIL import Image, ImageDraw, ImageFont

def research_list(request):
    search_query = request.GET.get('title', '')
    research_list = Research.objects.all()
    if search_query:
        research_list = research_list.filter(title__icontains=search_query)
    paginator = Paginator(research_list, 3)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'researches/research_list.html', {'page_obj': page_obj, 'search_query': search_query})

def research_detail(request, id):
    research = get_object_or_404(Research, id=id)
    media_files = Media_file.objects.filter(research=id)  # CamelCase для имен классов
    for file in media_files:
        file.is_video = file.media_file.url.lower().endswith(('.mp4', '.webm'))
    return render(request, 'researches/research_detail.html', {
        'research': research,
        'media_files': media_files
    })

def create_research(request):
    if request.method == 'POST':
        form = ResearchForm(request.POST)
        if form.is_valid():
            form.save()
            # Перенаправление на другую страницу после успешного создания
            return redirect('research_list')
    else:
        form = ResearchForm()

    context = {
        'form': form
    }
    return render(request, 'researches/research_form.html', context)


def media_file_create(request, research_id):
    research = get_object_or_404(Research, id=research_id)
    if request.method == 'POST':
        form = MediaFileForm(request.POST, request.FILES)
        if form.is_valid():
            media_file = form.save(commit=False)
            media_file.research = research
            media_file.save()
            return redirect('research_detail', id=research_id)
    else:
        form = MediaFileForm()
    return render(request, 'researches/media_file_form.html', {'form': form, 'research': research})

def media_file_detail(request, media_file_id):
    media_file = get_object_or_404(Media_file, id=media_file_id)
    # Определение, является ли файл видео
    is_video = media_file.media_file.url.lower().endswith(('.mp4', '.mov'))
    return render(request, 'researches/media_file_detail.html', {
        'media_file': media_file,
        'is_video': is_video  # Передаем информацию, является ли файл видео, в шаблон
    })

def delete_media_file(request, media_file_id):
    media_file = get_object_or_404(Media_file, id =media_file_id)
    research_id = media_file.research.id
    media_file.delete()
    return redirect('research_detail', id=research_id)


def media_file_pose(request, media_file_id):
    media_file = get_object_or_404(Media_file, id=media_file_id)
    return render(request, 'researches/media_file_pose.html', {
        'media_file': media_file,
     })


def media_file_mask(request, media_file_id):
    media_file = get_object_or_404(Media_file, id=media_file_id)
    return render(request, 'researches/media_file_mask.html', {
        'media_file': media_file,
     })

def media_file_yolo(request, media_file_id):
    media_file = get_object_or_404(Media_file, id=media_file_id)
    return render(request, 'researches/media_file_yolo.html', {
        'media_file': media_file,
     })


def media_file_filter_opencv(request, id):
    # Функция, генерирующая поток данных изображения.
    def generate(media_file_id, filter_type='RGB'):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        media_file_path = media_file.media_file.path
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
        video_path = video.media_file.path
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
        media_file_path = media_file.media_file.path
        cap = cv2.VideoCapture(media_file_path)

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
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

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
        media_file_path = media_file.media_file.path
        cap = cv2.VideoCapture(media_file_path)
        while True:
            success, img = cap.read()

            _, encoded_frame = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
        cv2.waitKey(50)
    return StreamingHttpResponse(stream_pose(id), content_type='multipart/x-mixed-replace; boundary=frame')

def media_file_YOLO_opencv(request, id):
    def stream_YOLO(media_file_id):
        media_file = get_object_or_404(Media_file, id=media_file_id)
        media_file_path = media_file.media_file.path
        cap = cv2.VideoCapture(media_file_path)
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
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

