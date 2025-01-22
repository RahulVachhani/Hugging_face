# speech/serializers.py

from rest_framework import serializers
from .models import  SpeechRecognitionResult



class SpeechResponseSerializer(serializers.Serializer):
    phonemes = serializers.ListField(child=serializers.CharField())
    audio = serializers.CharField()  # audio as base64-encoded string or file

class SpeechRecognitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeechRecognitionResult
        fields = ['id', 'audio_file', 'recognized_text', 'created_at']
        read_only_fields = ['id', 'recognized_text', 'created_at']
