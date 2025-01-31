

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

import random
import shutil
import tempfile
from django.conf import settings
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
from .models import GeneratedAudio, GeneratedVideo, SpeechRecognitionResult, GeneretedText, GeneratedImage, YoloImage
from . import hugging_face
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files import File
import cv2
from huggingface_hub import InferenceClient





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

        
        audio_file_name = "generated_audio.wav" 

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




pipe = hugging_face.model()

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
    


imgmodel, processor = hugging_face.model1() 


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


model4 = hugging_face.model2()

class ImageFaceDetectionAPI(APIView):
    def post(self, request, *args, **kwargs):
        # Ensure an image file is provided
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
        
            # Open the uploaded image
            image = Image.open(image_file)

            # Perform inference
            output = model4(image)
            results = Detections.from_ultralytics(output[0])

            # Create the annotator
            annotator = BoxAnnotator()

            # Annotate the image with the detections
            annotated_image = annotator.annotate(scene=image, detections=results)

            # Save annotated image to a BytesIO object to send in the response
            img_byte_arr = BytesIO()
            annotated_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Convert the annotated image into a Django InMemoryUploadedFile
            img_file = InMemoryUploadedFile(
                img_byte_arr, None, "annotated_image.png", 'image/png', img_byte_arr.getbuffer().nbytes, None
            )
        
            img = GeneratedImage.objects.create(
                img = img_file
            )

            return Response({
                'image_url': img.img.url  
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MediaFaceDetectionAPI(APIView):
    def post(self, request, *args, **kwargs):
        media_file = request.FILES.get('file')
        if not media_file:
            return Response({"error": "File is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Identify file type by extension
        file_name = media_file.name.lower()
        is_image = file_name.endswith(('.png', '.jpg', '.jpeg'))
        is_video = file_name.endswith(('.mp4', '.avi', '.mov'))

        if not is_image and not is_video:
            return Response({"error": "Unsupported file type. Use image or video."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if is_image:
                # Handle image file
                image = Image.open(media_file)
                output = model4(image)  # Perform inference
                results = Detections.from_ultralytics(output[0])

                annotator = BoxAnnotator()
                annotated_image = annotator.annotate(scene=image, detections=results)

                # Save the annotated image to a BytesIO object
                img_byte_arr = BytesIO()
                annotated_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Convert to Django InMemoryUploadedFile for saving
                img_file = InMemoryUploadedFile(
                    img_byte_arr, None, "annotated_image.png", 'image/png', img_byte_arr.getbuffer().nbytes, None
                )

                img = GeneratedImage.objects.create(img=img_file)

                return Response({'image_url': img.img.url}, status=status.HTTP_200_OK)

            elif is_video:
                # Handle video file
                temp_video_path = "/tmp/input_video.mp4"
                with open(temp_video_path, "wb") as f:
                    for chunk in media_file.chunks():
                        f.write(chunk)

            # Open the video using OpenCV
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    return Response({"error": "Invalid video file"}, status=status.HTTP_400_BAD_REQUEST)

            # Prepare to write the annotated video
                output_path = "/tmp/annotated_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )

                # Annotate video frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to PIL Image for annotation
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    output = model4(frame_pil)  
                    results = Detections.from_ultralytics(output[0])

                    annotator = BoxAnnotator()

                    annotated_frame = cv2.cvtColor(
                        np.array(annotator.annotate(scene=frame_pil, detections=results)), cv2.COLOR_RGB2BGR
                    )

                    out.write(annotated_frame)
                      
                cap.release()
                out.release()

                # Save annotated video to the database
                with open(output_path, "rb") as f:
                    video_file = File(f, name="annotated_video.mp4")
                    annotated_video = GeneratedVideo.objects.create(video=video_file)

                return Response({
                    "video_url": annotated_video.video.url  # URL of the saved video
                }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)






# Load the YOLO model
model_path = '/home/rahul/Documents/Yolo/runs/detect/train/weights/best.pt'
model = YOLO(model_path)



# Dictionary to store generated colors for each class index
class_color_map = {
    'motor_bike': (255, 0, 0),  # Red for bikes
    'with_helmet': (255, 255, 0),  # Yellow for helmets
    'without_helmet': (0, 0, 255),  # Blue for without helmet
}

class YoloPredict(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)
    
            # Open image and make predictions
        try:
            image = Image.open(image_file)
            # output = model(image)  # Perform inference
            # results = Detections.from_ultralytics(output[0])
            results = model(image)  # Perform inference
            # annotator = BoxAnnotator()
            # annotated_image = annotator.annotate(scene=image, detections=results)

            # Extract the results (boxes, labels, and confidence scores)
            boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box coordinates
            labels = results[0].boxes.cls.cpu().numpy()  # Class labels
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

            class_names = model.names 
            print(class_names)
            print(boxes)
            print('labels :',labels)
            print(type(labels))
            print(labels[1])
            print(confidences)
            # Convert the image to a NumPy array (RGB)
            annotated_image = np.array(image)

            # Initialize ImageDraw object to draw on the PIL Image
            annotated_image = Image.fromarray(annotated_image)  # Convert back to PIL Image for drawing
            draw = ImageDraw.Draw(annotated_image)

            # Load a font (optional: for adding text to boxes)
            font = ImageFont.load_default()  # Ensure it's for PIL Image, not NumPy array

            # Loop through the detections and annotate the image
            for i in range(len(boxes)):
                x1, y1, w, h = boxes[i]
                x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
                label_idx = int(labels[i])  # Get the class index
                label_name = class_names[label_idx]  # Get the class name
                confidence = confidences[i]
                text = f"{label_name} {confidence:.2f}"  # Use class name instead of numeric index

                 # Assign a unique color to the label
                if label_name in class_color_map:
                    color = class_color_map[label_name]  # Get the predefined color for the class
                else:
                    color = (255, 255, 255) 
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

                # Draw label and confidence
                draw.text((x1, y1), text, fill=color, font=font)
            
            img_byte_arr = BytesIO()
            annotated_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

                    # Convert to Django InMemoryUploadedFile for saving
            img_file = InMemoryUploadedFile(
                img_byte_arr, None, "annotated_image.png", 'image/png', img_byte_arr.getbuffer().nbytes, None
            )

            img = YoloImage.objects.create(img=img_file)
           
            return Response({'image_url': img.img.url}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
import os
class YoloPredictBikeDetectionVideo(APIView):
    def post(self, request, *args, **kwargs):
        video_file = request.FILES.get('video')
        if not video_file:
            return Response({"error": "Video file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Save the uploaded video to a temporary file
            video_data = video_file.read()
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_file.write(video_data)
            temp_video_file.close()  # Close the file to allow OpenCV to access it

            # Open the video file using OpenCV
            video_capture = cv2.VideoCapture(temp_video_file.name)
            if not video_capture.isOpened():
                os.remove(temp_video_file.name)  # Clean up the temporary file
                return Response({"error": "Failed to open video file."}, status=status.HTTP_400_BAD_REQUEST)

            # Get video properties
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            # Prepare for saving the video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
            output_video_path = "/tmp/input_video.mp4"  # Output video path
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break  # End of video

                # Convert frame to PIL image for annotation
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Perform inference
                results = model(pil_image)  # Perform inference

                # Extract the results (boxes, labels, and confidence scores)
                boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box coordinates
                labels = results[0].boxes.cls.cpu().numpy()  # Class labels
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

                class_names = model.names  # Get class names

                # Convert the PIL image back to NumPy array (RGB)
                annotated_image = np.array(pil_image)
                annotated_image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(annotated_image)

                # Load a font for adding text
                font = ImageFont.load_default()

                # Loop through the detections and annotate the image
                for i in range(len(boxes)):
                    x1, y1, w, h = boxes[i]
                    x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
                    label_idx = int(labels[i])  # Get the class index
                    label_name = class_names[label_idx]  # Get the class name
                    confidence = confidences[i]
                    text = f"{label_name} {confidence:.2f}"

                    # Assign color based on the class
                    if label_name in class_color_map:
                        color = class_color_map[label_name]
                    else:
                        color = (255, 255, 255)  # Default to white

                    # Draw bounding box and label
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, y1), text, fill=color, font=font)

                # Convert annotated image back to BGR for OpenCV
                annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

                # Write the annotated frame to the output video
                out_video.write(annotated_frame)

            # Release video capture and writer objects
            video_capture.release()
            out_video.release()

            # Clean up the temporary file
            os.remove(temp_video_file.name)

            # Create an InMemoryUploadedFile for the annotated video
            with open(output_video_path, 'rb') as f:
                video_content = ContentFile(f.read())
                video_file = InMemoryUploadedFile(video_content, None, "annotated_video.mp4", 'video/mp4', video_content.size, None)

            # Save the video file to the model
            video_instance = GeneratedVideo.objects.create(video=video_file)

            return Response({'video_url': video_instance.video.url}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# class YoloPredictBikeDetectionVideo(APIView):
#     def post(self, request, *args, **kwargs):
#         video_file = request.FILES.get('video')
#         if not video_file:
#             return Response({"error": "Video file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
#         try:
#             # Save the uploaded video to a temporary file
#             video_data = video_file.read()
#             temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#             temp_video_file.write(video_data)
#             temp_video_file.close()  # Close the file to allow OpenCV to access it

#             # Open the video file using OpenCV
#             video_capture = cv2.VideoCapture(temp_video_file.name)
#             if not video_capture.isOpened():
#                 os.remove(temp_video_file.name)  # Clean up the temporary file
#                 return Response({"error": "Failed to open video file."}, status=status.HTTP_400_BAD_REQUEST)

#             # Get video properties
#             frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = video_capture.get(cv2.CAP_PROP_FPS)

#             # Prepare for saving the video
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
#             output_video_path = "/tmp/annotated_video.mp4"  # Output video path
#             out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             while True:
#                 ret, frame = video_capture.read()
#                 if not ret:
#                     break  # End of video

#                 # Convert frame to PIL image for annotation
#                 pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                 # Perform inference
#                 results = model(pil_image)  # Perform inference

#                 # Extract the results (boxes, labels, and confidence scores)
#                 boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box coordinates
#                 labels = results[0].boxes.cls.cpu().numpy()  # Class labels
#                 confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

#                 class_names = model.names  # Get class names

#                 # Convert the PIL image back to NumPy array (RGB)
#                 annotated_image = np.array(pil_image)
#                 annotated_image = Image.fromarray(annotated_image)
#                 draw = ImageDraw.Draw(annotated_image)

#                 # Load a font for adding text
#                 font = ImageFont.load_default()

#                 # Loop through the detections and annotate the image
#                 for i in range(len(boxes)):
#                     x1, y1, w, h = boxes[i]
#                     x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
#                     label_idx = int(labels[i])  # Get the class index
#                     label_name = class_names[label_idx]  # Get the class name
#                     confidence = confidences[i]
#                     text = f"{label_name} {confidence:.2f}"

#                     # Draw bounding box and label
#                     draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
#                     draw.text((x1, y1), text, fill="red", font=font)

#                 # Convert annotated image back to BGR for OpenCV
#                 annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

#                 # Write the annotated frame to the output video
#                 out_video.write(annotated_frame)

#             # Release video capture and writer objects
#             video_capture.release()
#             out_video.release()

#             # Clean up the temporary file
#             os.remove(temp_video_file.name)

#             # Save the annotated video to the media directory
#             final_output_path = os.path.join(settings.MEDIA_ROOT, "annotated_video.mp4")
#             shutil.move(output_video_path, final_output_path)

#             # Create the file object for the annotated video
#             with open(final_output_path, 'rb') as f:
#                 video_content = ContentFile(f.read())
#                 video_file = InMemoryUploadedFile(video_content, None, "annotated_video.mp4", 'video/mp4', video_content.size, None)

#             # Save the video file to the model
#             video_instance = GeneratedVideo.objects.create(video=video_file)

#             # Return the URL of the annotated video
#             video_url = video_instance.video.url
#             return Response({'video_url': video_url}, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class TextToImageAPIview(APIView):
    def post(self, request):
        text = request.data.get('text','')
        if not text:
            return Response({'error':'Text is required'},status=status.HTTP_400_BAD_REQUEST)
        
        try:
            client = InferenceClient("black-forest-labs/FLUX.1-dev", token="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")
            print('----------------------------client done------------')
            image = client.text_to_image(text)
            print('----------------------------Image generated done------------')

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')  # Save the image in PNG format to the buffer
            img_byte_arr.seek(0)  # Reset the buffer's position to the beginning

            # Convert the BytesIO object to an InMemoryUploadedFile for saving to Django
            img_file = InMemoryUploadedFile(
                img_byte_arr, None, "generated_image.png", 'image/png', img_byte_arr.getbuffer().nbytes, None
            )

            # Save the image to the database
            img = GeneratedImage.objects.create(img=img_file)
            print('----------------------------Image saved to database------------')

        
            return Response({'image': img.img.url}, status=status.HTTP_200_OK)
        

        except Exception as e:
            return Response({'error': str(e)},status=status.HTTP_500_INTERNAL_SERVER_ERROR)
