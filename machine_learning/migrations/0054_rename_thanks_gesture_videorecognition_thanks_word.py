# Generated by Django 5.0 on 2024-01-05 09:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0053_alter_videorecognition_analysis_score'),
    ]

    operations = [
        migrations.RenameField(
            model_name='videorecognition',
            old_name='thanks_gesture',
            new_name='thanks_word',
        ),
    ]