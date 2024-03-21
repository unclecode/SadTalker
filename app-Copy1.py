import io
import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import torch

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

app = FastAPI()

# Initialize the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sadtalker_paths = init_path("./checkpoints", "./src/config", 256, False, "crop")
preprocess_model = CropAndExtract(sadtalker_paths, device)
audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

class GenerateRequest(BaseModel):
    driven_audio: UploadFile = File(..., media_type='audio/wav')
    source_image: UploadFile = File(..., media_type='image/png')
    pose_style: Optional[int] = 0
    batch_size: Optional[int] = 2
    expression_scale: Optional[float] = 1.0

@app.post("/generate")
async def generate_video(
    driven_audio: UploadFile = File(..., media_type='audio/wav'),
    source_image: UploadFile = File(..., media_type='image/png'),
    ref_eyeblink: UploadFile = None,
    pose_style: int = 0,
    batch_size: int = 2,
    expression_scale: float = 1.0
):
    # Save uploaded files
    with open("temp_audio.wav", "wb") as audio_file:
        audio_file.write(await driven_audio.read())
    with open("temp_image.png", "wb") as image_file:
        image_file.write(await source_image.read())

    # Generate video
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate("temp_image.png", first_frame_dir, "crop", source_image_flag=True, pic_size=256)

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(ref_eyeblink.filename)[-1]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        with open("temp_ref_eyeblink.mp4", "wb") as ref_eyeblink_file:
            ref_eyeblink_file.write(await ref_eyeblink.read())
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate("temp_ref_eyeblink.mp4", ref_eyeblink_frame_dir, "crop", source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    batch = get_data(first_coeff_path, "temp_audio.wav", device, ref_eyeblink_coeff_path, still=False)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, "temp_audio.wav", 
                                batch_size, input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                                expression_scale=expression_scale, still_mode=False, preprocess="crop", size=256)
    
    result = animate_from_coeff.generate(data, save_dir, "temp_image.png", crop_info, 
                                enhancer=None, background_enhancer=None, preprocess="crop", img_size=256)
    
    # Clean up temporary files
    os.remove("temp_audio.wav")
    os.remove("temp_image.png")
    if ref_eyeblink is not None:
        os.remove("temp_ref_eyeblink.mp4")

    return FileResponse(result, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)