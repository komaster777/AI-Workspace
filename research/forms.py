from django import forms
from .models import *
from django_select2.forms import ModelSelect2MultipleWidget
class ResearchForm(forms.ModelForm):
    class Meta:
        model = Research
        fields = ['title', 'description', 'technologies']  # Пример поля technologies
        widgets = {
            'technologies': forms.SelectMultiple(attrs={'class': 'django-select2'}),
        }
class MediaFileForm(forms.ModelForm):
    class Meta:
        model = Media_file
        fields = ['title','media_file', 'description']
        labels = {
            'media_file': 'Медиа файл объекта',
            'description': 'Описание изображения'
        }
        widgets = {
            'description': forms.Textarea(attrs={'cols': 80, 'rows': 3}),
        }
