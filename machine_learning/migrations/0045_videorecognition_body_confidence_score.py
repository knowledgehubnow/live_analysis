# Generated by Django 4.2.6 on 2023-12-20 09:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0044_videorecognition_voice_modulation_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='videorecognition',
            name='body_confidence_score',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
