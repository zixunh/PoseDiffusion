#!/bin/bash
cd /home/vision/zixun/PoseDiffusion/pose_diffusion

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
# python demo.py image_folder="samples/apple" ckpt="/home/vision/zixun/PoseDiffusion/ckpt/co3d_model_Apr16.pth"
python demo.py image_folder="/data/zixunh/co3d/data/subset_co3d_v2/toytruck/346_36113_66551/images" ckpt="/home/vision/zixun/PoseDiffusion/ckpt/co3d_model_Apr16.pth"
