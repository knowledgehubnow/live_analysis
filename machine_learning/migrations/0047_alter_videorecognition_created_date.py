# Generated by Django 5.0 on 2023-12-26 09:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0046_videorecognition_created_date'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videorecognition',
            name='created_date',
            field=models.DateTimeField(blank=True, editable=False, null=True),
        ),
    ]
