import io
import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import torch
import requests

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from config import MODEL_ID, VOICE_SETTINGS

app = FastAPI()

# Initialize the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sadtalker_paths = init_path("./checkpoints", "./src/config", 256, False, "crop")
preprocess_model = CropAndExtract(sadtalker_paths, device)
audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

CHUNK_SIZE = 1024
API_KEY = "e280cab1ce493a8626719c06793144e2" # os.environ["ELEVENLABS_API_KEY"]

def generate_speaking_avatar(driven_audio_path, source_image_path, ref_eyeblink_path=None, pose_style=0, batch_size=2, expression_scale=1.0):
    # Generate video
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(source_image_path, first_frame_dir, "crop", source_image_flag=True, pic_size=256)

    if ref_eyeblink_path is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.basename(ref_eyeblink_path))[-1]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink_path, ref_eyeblink_frame_dir, "crop", source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    batch = get_data(first_coeff_path, driven_audio_path, device, ref_eyeblink_coeff_path, still=False)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, driven_audio_path,
                               batch_size, input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                               expression_scale=expression_scale, still_mode=False, preprocess="crop", size=256)

    result = animate_from_coeff.generate(data, save_dir, source_image_path, crop_info,
                               enhancer=None, background_enhancer=None, preprocess="crop", img_size=256)

    return result

@app.post("/generate")
async def generate_video(
    driven_audio: UploadFile = File(None, media_type='audio/wav'),
    source_image: UploadFile = File(..., media_type='image/png'),
    text_input: str = Form(None),
    voice_id: str = Form("your_default_voice_id"),
    ref_eyeblink: UploadFile = None,
    pose_style: int = Form(0),  # Change this to Form
    batch_size: int = Form(2),  # Change this to Form
    expression_scale: float = Form(1.0),  # Change this to Form
):
    print("Batch Size:", batch_size)
    # Save uploaded files
    with open("temp_image.png", "wb") as image_file:
        image_file.write(await source_image.read())

    if driven_audio is not None:
        with open("temp_audio.wav", "wb") as audio_file:
            audio_file.write(await driven_audio.read())
        driven_audio_path = "temp_audio.wav"
    elif text_input is not None and voice_id is not None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": API_KEY
        }
        data = {
            "text": text_input,
            "model_id": MODEL_ID,
            "voice_settings": VOICE_SETTINGS
        }
        response = requests.post(url, json=data, headers=headers)
        with open('temp_audio.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        driven_audio_path = "temp_audio.mp3"
    else:
        raise ValueError("Either driven_audio or text_input and voice_id should be provided.")

    if ref_eyeblink is not None:
        with open("temp_ref_eyeblink.mp4", "wb") as ref_eyeblink_file:
            ref_eyeblink_file.write(await ref_eyeblink.read())
        ref_eyeblink_path = "temp_ref_eyeblink.mp4"
    else:
        ref_eyeblink_path = None

    result = generate_speaking_avatar(driven_audio_path, "temp_image.png", ref_eyeblink_path, pose_style, batch_size, expression_scale)

    # Clean up temporary files
    os.remove(driven_audio_path)
    os.remove("temp_image.png")
    if ref_eyeblink is not None:
        os.remove("temp_ref_eyeblink.mp4")

    return FileResponse(result, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)