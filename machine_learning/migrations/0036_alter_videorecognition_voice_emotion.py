# Generated by Django 4.2.6 on 2023-12-16 05:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0035_alter_videorecognition_filler_words_used_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videorecognition',
            name='voice_emotion',
            field=models.JSONField(blank=True, default=None, null=True),
        ),
    ]
