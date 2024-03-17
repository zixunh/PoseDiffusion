# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.9, CUDA 11.6

conda deactivate

# Set environment variables
export ENV_NAME=posediffusion_zixun
export PYTHON_VERSION=3.9

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
wget https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl
python -m pip install pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl

# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom 
pip install accelerate==0.24.0

# Install HLoc for extracting 2D matches (optional if GGS is not needed); pycolmap>=0.6.0
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc

cd dependency/hloc
python -m pip install -e .
cd ../../

# Ensure the version of pycolmap is not 0.5.0
# pip install --upgrade "pycolmap>=0.3.0,<=0.4.0"

