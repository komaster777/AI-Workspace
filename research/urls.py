from django.urls import path
from .views import *
from . import views

urlpatterns = [
    path('', views.research_list, name='research_list'),
    path('research/<int:id>/', views.research_detail, name='research_detail'),
    path('research/create/', views.create_research, name='create_research'),
    path('media-file/create/<int:research_id>/', views.media_file_create, name='media_file_create'),
    path('media-file/detail/<int:media_file_id>/', views.media_file_detail, name='media_file_detail'),
    path('media-file/delete/<int:media_file_id>/', views.delete_media_file, name='delete_media_file'),
    path('media_file_mask/mask/<int:media_file_id>/', views.media_file_mask, name='media_file_mask'),
    path('media_file_pose/pose/<int:media_file_id>/', views.media_file_pose, name='media_file_pose'),
    path('media_file_yolo/yolo/<int:media_file_id>/', views.media_file_yolo, name='media_file_yolo'),
    path('media_file_filter_opencv/<int:id>/', media_file_filter_opencv, name='media_file_filter_opencv'),
    path('media_file_mask_opencv/<int:id>/', media_file_mask_opencv, name='media_file_mask_opencv'),
    path('media_file_contours_opencv/<int:id>/', media_file_contours_opencv, name='media_file_contours_opencv'),
    path('media_file_pose_opencv/<int:id>/', media_file_pose_opencv, name='media_file_pose_opencv'),
    path('media_file_YOLO_opencv/<int:id>/', media_file_YOLO_opencv, name='media_file_YOLO_opencv'),
]









