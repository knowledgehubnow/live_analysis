from django.db import models
from django.conf import settings
import ast
import json
# Create your models here.


class VideoRecognition(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    analysis_score = models.FloatField(default=0.0)
    word_per_minute = models.FloatField(null=True, blank=True)
    language_analysis = models.JSONField(null=True, blank=True,default=None)
    voice_modulation_analysis = models.JSONField(null=True, blank=True, default=None)
    energy_level_analysis = models.CharField(max_length=255, null=True, blank=True)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    filler_words_used = models.JSONField(null=True, blank=True)
    frequently_used_word = models.JSONField(null=True, blank=True)
    voice_emotion = models.JSONField(null=True, blank=True, default=None)
    confidence = models.CharField(max_length=100,null=True, blank=True)
    eye_bling = models.CharField(max_length=100,null=True, blank=True)
    hand_movement = models.CharField(max_length=100,null=True, blank=True)
    eye_contact = models.CharField(max_length=100,null=True, blank=True)
    thanks_gesture = models.CharField(max_length=100,null=True, blank=True)
    greeting = models.CharField(max_length=100,null=True, blank=True)
    greeting_gesture = models.CharField(max_length=100,null=True, blank=True)
    voice_tone = models.CharField(max_length=100,null=True, blank=True)
    voice_pauses = models.CharField(max_length=100,null=True, blank=True)
    appropriate_facial = models.CharField(max_length=100,null=True, blank=True)
    body_posture = models.CharField(max_length=100,null=True, blank=True)

    def __str__(self):
        return self.name
        
    

    
