import numpy as np

# Load the NPZ file
data = np.load('/home/vision/zixun/PoseDiffusion/pose_diffusion/samples/apple/gt_cameras.npz')
print(data['gtR'].shape, data['gtT'].shape, data['gtFL'].shape)