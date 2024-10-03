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
import random
import json
import torch

from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from tools.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from tools.camera_utils import cameraList_from_camInfos, camera_to_JSON
from tools.graphics_utils import get_all_px_dir

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.split = args.split
        load_depth = args.load_depth
        load_normal = args.load_normal
        load_mask = args.load_mask

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold, args.ratio, split=self.split, load_depth=load_depth, load_normal=load_normal, load_mask=load_mask, normal_folder=args.normal_folder, depth_folder=args.depth_folder)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        
        self.trans = scene_info.trans
        self.scale = scene_info.scale

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        gaussians.extent = self.cameras_extent
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
            for idx, camera in enumerate(self.train_cameras[resolution_scale] + self.test_cameras[resolution_scale]):
                camera.idx = idx

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        if args.depth_type == "traditional":
            self.dirs = None
        elif args.depth_type == "intersection":
            self.dirs = get_all_px_dir(self.getTrainCameras()[0].intr, self.getTrainCameras()[0].image_height, self.getTrainCameras()[0].image_width).cuda()
        self.first_name = scene_info.first_name

    def save(self, iteration, visi=None, surf=None, save_splat=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_inside_ply(os.path.join(point_cloud_path, "point_cloud_inside.ply"))
        
        if visi is not None:
            self.gaussians.save_visi_ply(os.path.join(point_cloud_path, "visi.ply"), visi)
        
        if surf is not None:
            self.gaussians.save_visi_ply(os.path.join(point_cloud_path, "surf.ply"), surf)
        
        if save_splat:
            self.gaussians.save_splat(os.path.join(point_cloud_path, "pcd.splat"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getFullCameras(self, scale=1.0):
        if self.split:
            return self.train_cameras[scale] + self.test_cameras[scale]
        else:
            return self.train_cameras[scale]
    
    def getUpCameras(self):
        return self.random_cameras_up
    
    def getAroundCameras(self):
        return self.random_cameras_around
    
    def getRandCameras(self, n, up=False, around=True, sample_mode='uniform'):
        if up and around:
            n = n // 2
        
        cameras = []
        if up:
            up_cameras = self.getUpCameras().copy()
            idx = torch.randperm(len(up_cameras))[: n]
            if n == 1:
                cameras.append(up_cameras[idx])
            else:
                cameras.extend(up_cameras[idx])
        if around:
            around_cameras = self.getAroundCameras()
            
            if sample_mode == 'random':
                idx = torch.randperm(len(around_cameras))[: n]
            elif sample_mode == 'uniform':
                idx = torch.arange(len(around_cameras))[::len(around_cameras)//n]
            else:
                assert False, f"Unknown sample_mode: {sample_mode}"
            
            if n == 1:
                cameras.append(around_cameras[idx])
            else:
                cameras.extend(around_cameras[idx])
        return cameras