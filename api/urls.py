from django.urls import path
from . import views

urlpatterns = [
    path('text-to-speech/', views.TextToSpeechView.as_view()),
    path('speech-to-text/', views.SpeechToTextAPIView.as_view()),
    path('image-to-text/', views.ImageToTextAPIview.as_view()),
    path('face-detection/', views.MediaFaceDetectionAPI.as_view())  
]
