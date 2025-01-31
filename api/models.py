# models.py
from django.db import models

class GeneratedAudio(models.Model):
    text = models.TextField()
    audio_file = models.FileField(upload_to='generated_audio/', null=True, blank=True)
    phonemes = models.TextField()

    def __str__(self):
        return f"Generated Audio for: {self.text[:30]}..."


class SpeechRecognitionResult(models.Model):
    audio_file = models.FileField(upload_to="uploaded_audio/")  # Path for user-uploaded audio
    recognized_text = models.TextField()  # Recognized text from audio
    created_at = models.DateTimeField(auto_now_add=True)



class GeneretedText(models.Model):
    img = models.ImageField(upload_to='images/')
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:20]
    

class GeneratedImage(models.Model):
    img = models.ImageField(upload_to='image/')


class GeneratedVideo(models.Model):
    video = models.FileField(upload_to="annotated_videos/")
    created_at = models.DateTimeField(auto_now_add=True)


class YoloImage(models.Model):
    img = models.ImageField(upload_to='yolo-image/')