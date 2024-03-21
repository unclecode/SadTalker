# export PYTHONPATH=/root/workspace/SadTalker:$PYTHONPATH
# python3.8 -m venv venv
# source venv/bin/activate
# Ask user to press enter to continue
read -p "Press enter to continue installing requirements"
python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python3.8 -m pip install fastapi uvicorn python-multipart
python3.8 -m pip install wheel setuptools
apt update
read -p "Press enter to continue installing ffmpeg"
apt install ffmpeg 
read -p "Press enter to continue installing requirements"
python3.8 -m pip install -r requirements.txt

# bash scripts/download_models.sh
# python3.8 inference.py --driven_audio ./examples/driven_audio/RD_Radio31_000.wav --source_image 'examples/source_image/art_0.png' --result_dir ./results
# curl -X POST   -H "Content-Type: multipart/form-data"   -F "driven_audio=@./examples/driven_audio/RD_Radio31_000.wav"   -F "source_image=@examples/source_image/art_0.png"   -F "pose_style=0"   -F "batch_size=2"   -F "expression_scale=1.0"   http://localhost:8000/generate   --output result.mp4