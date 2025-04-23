from django.db import models

class Class(models.Model):
    title = models.CharField(max_length=100)
    title_russian = models.CharField(max_length=100, blank=True)
    def __str__(self):
        return self.title

class Technology(models.Model):
    title = models.CharField(max_length= 100)
    description = models.TextField('Описание технологии',blank=True)
class Research(models.Model):
    title = models.CharField('Название исследования',max_length=100)
    date_collected = models.DateField(auto_now_add=True)
    description = models.TextField('Описание исследования',blank=True)
    technologies = models.ManyToManyField(Technology, related_name='researches')
    def __str__(self):
        return self.title

class Media_file(models.Model):
    title = models.CharField('Название объекта', max_length=100, blank=True)
    media_file = models.FileField('Медиа файл объекта',upload_to='photos/')
    research = models.ForeignKey(Research, on_delete=models.CASCADE, verbose_name='Исследование')
    description = models.TextField('Описание изображения',blank=True)

    def __str__(self):
        return f"Photo of {self.research.title}"

class PhotoClass(models.Model):
    photo = models.ForeignKey(Media_file, on_delete=models.CASCADE, related_name='detected_classes')
    type = models.ForeignKey(Class, on_delete=models.CASCADE, related_name='type')
    count = models.IntegerField()

class ResearchTechnology(models.Model):
    researches = models.ForeignKey(Research, on_delete=models.CASCADE, related_name='detected_classes')
    technologies = models.ForeignKey(Technology, on_delete=models.CASCADE, related_name='technologies')

class Object_research(models.Model):
    title = models.CharField(max_length=100)