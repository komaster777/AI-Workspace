# Generated by Django 5.0.2 on 2024-03-06 13:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('research', '0004_rename_classes_class_rename_researches_research_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='research',
            name='technologies',
            field=models.ManyToManyField(related_name='researches', to='research.technology'),
        ),
    ]
