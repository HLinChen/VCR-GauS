env=vcr
conda create -n $env -y python=3.10
conda activate $env
pip install -e ".[train]"
export CUDA_HOME=/usr/local/cuda-11.2
pip install -r requirements.txt