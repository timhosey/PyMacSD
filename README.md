brew install cmake protobuf git wget python

python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

export PYTORCH_ENABLE_MPS_FALLBACK=1# PyMacSD
