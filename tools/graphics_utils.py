#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))   # w2c
    Rt[:3, :3] = R.transpose()      # w2c
    Rt[:3, 3] = t                   # w2c
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)         # c2w
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)         # w2c
    return np.float32(Rt)

def getView2World(R, t):
    '''
    R: w2c
    t: w2c
    '''
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()      # c2w
    Rt[:3, 3] = -R.transpose() @ t  # c2w
    Rt[3, 3] = 1.0

    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    '''
    normalized intrinsics
    '''
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getIntrinsic(fovX, fovY, h, w):
    focal_length_y = fov2focal(fovY, h)
    focal_length_x = fov2focal(fovX, w)
    
    intrinsic = np.eye(3)
    intrinsic = torch.eye(3, dtype=torch.float32)
    
    intrinsic[0, 0] = focal_length_x # FovX
    intrinsic[1, 1] = focal_length_y # FovY
    intrinsic[0, 2] = w / 2
    intrinsic[1, 2] = h / 2
    
    return intrinsic


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device).add_(0.5) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device).add_(0.5) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing='ij')
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz


def depth2point(depth_image, intrinsic_matrix, extrinsic_matrix):
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_cam.reshape(*depth_image.shape, 3), xyz_world.reshape(*depth_image.shape, 3)


@torch.no_grad()
def get_all_px_dir(intrinsics, height, width):
    """
    # Calculate the view direction for all pixels/rays in the image.
    # This is used for intersection calculation between ray and voxel textures.
    # """

    a, ray_dir = depth2point_cam(torch.ones(1, 1, 1, height, width).cuda(), intrinsics[None])
    a, ray_dir = a.squeeze(), ray_dir.squeeze()
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
    
    ray_dir = ray_dir.permute(2, 0, 1) # 3, H, W
    return ray_dir