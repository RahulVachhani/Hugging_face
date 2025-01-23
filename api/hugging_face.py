import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pickle

def model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def model1():
    with open("models/imagetotext.sav", "rb") as f:
        model_data = pickle.load(f)

    # Now you can access the model and processor
    model = model_data['model']
    processor = model_data['processor']

    return model,processor



def model2():
    with open('models/faceDetection.pkl', 'rb') as f:
        model_data = pickle.load(f)

    return model_data