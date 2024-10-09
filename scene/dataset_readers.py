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

import os
import sys
import cv2
import json
import numpy as np
import open3d as o3d
from PIL import Image, ImageFile
from pathlib import Path
from typing import NamedTuple
from plyfile import PlyData, PlyElement
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from tools.graphics_utils import getWorld2View2, focal2fov, fov2focal
from tools.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tools.math_utils import normalize_pts
from process_data.convert_data_to_json import bound_by_points

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: None
    normal: None
    mask: None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    trans: np.array
    scale: np.array
    first_name: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_depth=False, load_normal=False, load_mask=False, normal_folder='normals', depth_folder='depths'):
    if load_depth:
        depths_folder = images_folder.replace('images', depth_folder)
    
    if load_normal:
        normals_folder = images_folder.replace('images', normal_folder)
    if load_mask:
        mask_folder = images_folder.replace('images', 'masks')
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        depth = None
        if load_depth:
            depth_path = os.path.join(depths_folder, os.path.basename(extr.name).replace('jpg', 'npz').replace('png', 'npz'))
            if os.path.exists(depth_path):
                depth = np.load(depth_path)['arr_0']
            else:
                depth_path = os.path.join(depths_folder, os.path.basename(extr.name).replace('jpg', 'png'))
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if depth.ndim == 2: depth = depth[..., None]
        
        normal = None
        if load_normal:
            normal_path = os.path.join(normals_folder, os.path.basename(extr.name).replace('png', 'npz').replace('jpg', 'npz').replace('JPG', 'npz'))
            normal = np.load(normal_path)['arr_0'] # -1, 1

        mask = None
        if load_mask:
            mask_path = os.path.join(mask_folder, os.path.basename(extr.name).replace('jpg', 'png'))
            mask_path = mask_path if os.path.exists(mask_path) else \
                        os.path.join(mask_folder, os.path.basename(extr.name)[1:])
            mask = Image.open(mask_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=depth, normal=normal, mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz) if normals is None else normals

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def get_inside_mask(pts, trans, scale):
    pts = normalize_pts(pts, trans, scale)
    
    inside = np.all(np.abs(pts) < 1.5, axis=-1)
    return inside

def filter_point_cloud(trans, scale, xyz, rgb, nb_points=5, radius=0.1):
    inside = get_inside_mask(xyz, trans, scale)
    xyz_inside = xyz[inside]
    rgb_inside = rgb[inside]
    xyz_outside = xyz[~inside]
    rgb_outside = rgb[~inside]
    
    pcd_inside = o3d.geometry.PointCloud()
    pcd_inside.points = o3d.utility.Vector3dVector(xyz_inside)
    pcd_inside.colors = o3d.utility.Vector3dVector(rgb_inside)
    
    pcd_inside_filter, ind = pcd_inside.remove_radius_outlier(nb_points, radius)
    
    xyz_inside = np.asarray(pcd_inside_filter.points)
    rgb_inside = np.asarray(pcd_inside_filter.colors)
    
    xyz = np.concatenate((xyz_inside, xyz_outside), axis=0)
    rgb = np.concatenate((rgb_inside, rgb_outside), axis=0)
    
    return xyz, rgb

def readColmapSceneInfo(path, images, eval, llffhold=8, ratio=0, split=False, load_depth=False, load_normal=False, load_mask=False, normal_folder='normals', depth_folder='depths'):
    colmap_dir = os.path.join(path, "sparse/0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(path, "sparse")
    try:
        cameras_extrinsic_file = os.path.join(colmap_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    ply_path = os.path.join(colmap_dir, "points3D.ply")
    bin_path = os.path.join(colmap_dir, "points3D.bin")
    txt_path = os.path.join(colmap_dir, "points3D.txt")
    
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), load_depth=load_depth, load_normal=load_normal, load_mask=load_mask, normal_folder=normal_folder, depth_folder=depth_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    meta_fname = f"{path}/meta.json"
    if os.path.exists(meta_fname):
        with open(meta_fname) as file:
            meta = json.load(file)
        trans = np.array(meta["trans"], dtype=np.float32)
        scale = np.array(meta["scale"], dtype=np.float32)
    else:
        print("No meta.json file found, using default values.")
        
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            xyz, rgb = filter_point_cloud(trans, scale, xyz, rgb)
        #     storePly(ply_path, xyz, rgb)
        # try:
        #     pcd = fetchPly(ply_path)
        # except:
        #     pcd = None
        
        trans, scale, bounding_box = bound_by_points(xyz)
        meta = {
            'trans': trans.tolist(),
            'scale': scale.tolist()
        }
        with open(meta_fname, "w") as file:
            json.dump(meta, file, indent=4)

    if ratio > 0:
        len_train = int(len(cam_infos) * ratio)
        llffhold = len(cam_infos) // len_train
        train_idx = set([int(i * llffhold) for i in range(len_train)])
        test_idx = set(range(len(cam_infos))) - train_idx
        train_cam_infos = [cam_infos[i] for i in train_idx]
        test_cam_infos = [cam_infos[i] for i in test_idx]
    elif eval:
        if split and "test" in meta:
            train_cam_infos = [c for c in cam_infos if c.image_name in meta["train"]]
            test_cam_infos = [c for c in cam_infos if c.image_name in meta["test"]]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        
    print(f"Train: {len(train_cam_infos)}, Test: {len(test_cam_infos)}")

    first_name = test_cam_infos[0].image_name if eval else cam_infos[0].image_name
    nerf_normalization = getNerfppNorm(train_cam_infos)

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz, rgb = filter_point_cloud(trans, scale, xyz, rgb)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           trans=trans,
                           scale=scale,
                           first_name=first_name)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}