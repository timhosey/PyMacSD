brew install cmake protobuf git wget python

python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

export PYTORCH_ENABLE_MPS_FALLBACK=1# PyMacSD

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

mkdir -p models/Stable-diffusion
curl -L -o models/Stable-diffusion/v1-5-pruned.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt

python main.py