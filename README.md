# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

![Teaser](https://raw.githubusercontent.com/posediffusion/posediffusion.github.io/main/resources/teaser.gif)

<p dir="auto">[<a href="https://arxiv.org/pdf/2306.15667.pdf" rel="nofollow">Paper</a>]
[<a href="https://posediffusion.github.io/" rel="nofollow">Project Page</a>]</p>

## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.9, PyTorch 1.13, and CUDA 11.6.
##### Python 3.9; CUDA 11.6; PyTorch 1.13.1; PyTorch3D 0.7.2 (prebuilt wheel)
```.bash
# This Script Assumes Python 3.9, CUDA 11.6
# download prebuilt wheel for pytorch3d first:
# [pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl](https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl)

conda deactivate

# Set environment variables
export ENV_NAME=posediffusion

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using pip and prebuilt wheel
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl
```
##### HLoc; pycolmap>=0.6.0
```
# Install pip packages
python -m pip install hydra-core --upgrade
python -m pip install omegaconf opencv-python einops visdom 
python -m pip install accelerate==0.24.0

# Install HLoc for extracting 2D matches (optional if GGS is not needed); pycolmap>=0.6.0
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc

cd dependency/hloc
python -m pip install -e .
```

## Quick Start

### 1. Download Checkpoint

Download the model checkpoint trained on Co3D from [Dropbox](https://www.dropbox.com/s/tqzrv9i0umdv17d/co3d_model_Apr16.pth?dl=0). The predicted camera poses and focal lengths are defined in [NDC coordinate](https://pytorch3d.org/docs/cameras).


### 2. Run the Demo

```.bash
# run visdom server
visdom
# run demo
python demo.py image_folder="samples/apple" ckpt="/PATH/TO/DOWNLOADED/CKPT"
```

You can experiment with your own data by specifying a different `image_folder`.

On a RTX 3080 Ti, the inference time for a 20-frame sequence is approximately 5.07 seconds without GGS and around 170.03 seconds with GGS (including matching extraction). 

On a TITAN RTX, the inference time for a 20-frame sequence is approximately 1.22 seconds (ARE: 3.06) without GGS and around 112.07 seconds (ARE: 2.15) with GGS; 22.21 seconds if start_step==1 (ARE: 2.38); 15.38 seconds if not optimize FL, R, and T separately (ARE: 2.58); 11.25 seconds for matching extraction (with commented GGS, 12.47 seconds in total).

You can choose to enable or disable GGS (or other settings) in `./cfgs/default.yaml`.

We use [Visdom](https://github.com/fossasia/visdom) by default for visualization. Ensure your Visdom settings are correctly configured to visualize the results accurately. However, Visdom is not necessary for running the model.

Umeyama's Algorithm is used to align the predicted camera poses onto groundtruth:
```
# 7dof alignment, using Umeyama's algorithm
pred_cameras_aligned = corresponding_cameras_alignment(
    cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
)
```

## Training

### 1. Preprocess Annotations

Start by following the instructions [here](https://github.com/amyxlase/relpose-plus-plus#pre-processing-co3d) to preprocess the annotations of the Co3D V2 dataset. This will significantly reduce data processing time during training.

Refer to [synsin](https://github.com/facebookresearch/synsin/blob/main/REALESTATE.md) for RealEstate10k. For small subset, refer to [AttnRend](https://github.com/yilundu/cross_attention_renderer/tree/master).

### 2. Specify Paths

Next, specify the paths for `CO3D_DIR` and `CO3D_ANNOTATION_DIR` in `./cfgs/default_train.yaml`. `CO3D_DIR` should be set to the path where your downloaded Co3D dataset is located, while `CO3D_ANNOTATION_DIR` should point to the location of the annotation files generated after completing the preprocessing in step 1.

### 3. Start Training

- For 1-GPU Training:
  ```bash
  python train.py
  ```

- For multi-GPU training, launch the training script using [accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch), e.g., training on 8 GPUs (processes) in 1 node (machines):
  ```bash
  accelerate launch --num_processes=8 --multi_gpu --num_machines=1 train.py 
  ```
  
All configurations are specified inside `./cfgs/default_train.yaml`. Please notice that we use Visdom to record logs.

## Testing

### 1. Specify Paths

Please specify the paths `CO3D_DIR`, `CO3D_ANNOTATION_DIR`, and `resume_ckpt` in `./cfgs/default_test.yaml`. The flag `resume_ckpt` refers to your downloaded model checkpoint.

### 2. Run Testing

```bash
python test.py
```

You can check different testing settings by adjusting `num_frames`, `GGS.enable`, and others in `./cfgs/default_test.yaml`.


## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [guided-diffusion](https://github.com/openai/guided-diffusion), [hloc](https://github.com/cvg/Hierarchical-Localization), [relpose](https://github.com/jasonyzhang/relpose).


## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

