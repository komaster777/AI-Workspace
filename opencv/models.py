from django.db import models


class Media_file(models.Model):
    title = models.CharField('Название объекта', max_length=100, blank=True)
    media_file = models.FileField('Медиа файл объекта',upload_to='photos/')
    description = models.TextField('Описание изображения',blank=True)
