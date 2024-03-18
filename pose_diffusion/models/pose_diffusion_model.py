# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding

import models
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras

logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(IMAGE_FEATURE_EXTRACTOR, _recursive_=False)
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
        training=True,
        batch_repeat=-1,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        shapelist = list(image.shape) # [B, frame_num, 3, H, W]
        batch_num = shapelist[0]
        frame_num = shapelist[1]

        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:]) # [B*frame_num, 3, H, W]
        z = self.image_feature_extractor(reshaped_image).reshape(batch_num, frame_num, -1) # [B, frame_num, 384]
        if training:
            pose_encoding = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

            if batch_repeat > 0:
                pose_encoding = pose_encoding.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding = pose_encoding.reshape(batch_num, -1, self.target_dim)

            diffusion_results = self.diffuser(pose_encoding, z=z)

            diffusion_results["pred_cameras"] = pose_encoding_to_camera(
                diffusion_results["x_0_pred"], pose_encoding_type=self.pose_encoding_type
            )

            return diffusion_results
        else:
            B, N, _ = z.shape # [B, frame_num, 384]

            target_shape = [B, N, self.target_dim] # [B, frame_num, 9 (denoiser.target_dim)]

            # sampling
            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                shape=target_shape, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step
            ) #--> target_shape

            # convert the encoded representation to PyTorch3D cameras
            # in: [B, N, (3+4+2)]; 3 for t; 4 for R; 2 for focal len;
            # out: PerspectiveCameras(focal_length=focal_length, R=R, T=abs_T, device=R.device);
            # focal_length: List[[fx, fy]];
            # R: List[{3x3}]; T: List[{1x3}];
            pred_cameras = pose_encoding_to_camera(pose_encoding, pose_encoding_type=self.pose_encoding_type)

            diffusion_results = {"pred_cameras": pred_cameras, "z": z}

            return diffusion_results
