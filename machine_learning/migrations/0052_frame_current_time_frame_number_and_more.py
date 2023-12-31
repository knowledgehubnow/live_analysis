# Generated by Django 5.0 on 2024-01-04 06:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0051_rename_video_detectedframes_posture'),
    ]

    operations = [
        migrations.AddField(
            model_name='frame',
            name='current_time',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='number',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='videorecognition',
            name='video_durations',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
