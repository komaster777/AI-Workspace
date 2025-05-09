import cv2
import numpy as np
import torch
import pickle
import time
import sys
import tkinter as tk
from tkinter import ttk
from Cython.Compiler.FlowControl import Unknown
from PIL import Image, ImageTk
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist
import mediapipe as mp
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
# print(torch.cuda.is_available())
# print(torch.__version__)
#
# model = YOLO("yolov8n-pose.pt")
# print(model.device)  # Должно быть 'cuda:0' если используется GPU

print("CUDA доступна:", torch.cuda.is_available())
print("Количество GPU:", torch.cuda.device_count())
print("Имя устройства:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Нет GPU")

print("CUDA доступна:", torch.cuda.is_available())
print("PyTorch устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model = YOLO("yolov8n-pose.pt")
model.to("cuda")  # Явно переносим на GPU
print("YOLO устройство:", model.device)

# Запускаем предсказание
results = model.predict(source=0, show=True, device='cuda')  # Важно указать device
