# # speech/views.py

# import base64
# from io import BytesIO
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# import torch
# from kokoro import generate
# from models1 import build_model
# from .models import GeneratedAudio

# class TextToSpeechView(APIView):
#     def post(self, request):
#         # Extract text from the request data
#         text = request.data.get('text', '')
#         if not text:
#             return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)

#         # Load the model and voicepack
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         MODEL = build_model('kokoro-v0_19.pth', device)
#         VOICE_NAME = 'bm_george'
#         VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)

#         # Generate speech from the text
#         audio, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])

#         # Convert audio to base64 encoded string
#         audio_base64 = base64.b64encode(audio).decode('utf-8')

#         generated_audio = GeneratedAudio.objects.create(
#             text=text,
#             audio_file=audio_base64,
#             phonemes=out_ps
#         )

#         # Serialize response
#         response_data = {
#             'phonemes': out_ps,
#             'audio': audio_base64  # Send audio as base64 string
#         }

#         # Return the serialized data
#         return Response(response_data, status=status.HTTP_200_OK)

import numpy as np
from io import BytesIO
from scipy.io.wavfile import write  # Import to save as wav
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
from .serializers import  SpeechRecognitionSerializer
from models.kokoro import generate
from models.models1 import build_model
from django.core.files.base import ContentFile
from .models import GeneratedAudio, SpeechRecognitionResult, GeneretedText
from .hugging_face import model,model1
from PIL import Image



class TextToSpeechView(APIView):
    def post(self, request):
        
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the model and voicepack
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL = build_model('models/kokoro-v0_19.pth', device)
        VOICE_NAME = 'bm_george'
        VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)

        # Generate speech from the text
        audio, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])

        # Convert audio to a proper WAV file
        audio_file_name = "generated_audio.wav"  # You can customize the name of the file

        # Ensure audio is in the correct format (if needed)
        if isinstance(audio, np.ndarray):
            # Normalize the audio if it's in float format and convert to int16
            audio = np.int16(audio * 32767)  # Convert to 16-bit PCM audio

        # Save audio as a WAV file in a BytesIO buffer
        audio_buffer = BytesIO()
        write(audio_buffer, 22050, audio)  # Use the correct sample rate (e.g., 22050 Hz)
        audio_buffer.seek(0)

        # Save to the GeneratedAudio model
        audio_file = ContentFile(audio_buffer.read(), name=audio_file_name)

        # Create the GeneratedAudio object
        generated_audio = GeneratedAudio.objects.create(
            text=text,
            phonemes=out_ps,
            audio_file=audio_file,  # Save the audio file
        )

        # Serialize response
        response_data = {
            'phonemes': out_ps,
            'audio': generated_audio.audio_file.url  # Send audio file URL
        }

        # Return the serialized data
        return Response(response_data, status=status.HTTP_200_OK)




pipe = model()

class SpeechToTextAPIView(APIView):
    def post(self, request, *args, **kwargs):
        
        audio_file = request.FILES.get('audio_file')
        print('hello')
        if not audio_file:
            return Response({"error": "Audio file is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            
            audio_content = audio_file.read()

           
            result = pipe(audio_content, return_timestamps=True)
            
            recognition_result = SpeechRecognitionResult.objects.create(
                audio_file=audio_file, 
                recognized_text=result["text"]  
            )

          
            serializer = SpeechRecognitionSerializer(recognition_result)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    


imgmodel, processor = model1() 


class ImageToTextAPIview(APIView):
      def post(self, request, *args, **kwargs):
        # Ensure an image file is provided
        image_file = request.FILES.get('img')
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
       
            image = Image.open(image_file.file) 
          
            # Preprocess the image
            inputs = processor(images=image, return_tensors="pt")

            # Generate a caption using the model
            outputs = imgmodel.generate(**inputs)

            # Decode the generated caption
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            GeneretedText.objects.create(
                img = image_file,
                text = caption
            )

            # Return the generated caption in the response
            return Response({"caption": caption}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


