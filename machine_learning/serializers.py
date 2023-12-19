from rest_framework import serializers
from .models import *

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = ('video_file',)

class VideoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"
    
class VideoDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"



