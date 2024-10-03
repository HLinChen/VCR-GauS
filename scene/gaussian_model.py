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
import torch
import numpy as np
from torch import nn
from copy import deepcopy
try:
    from simple_knn._C import distCUDA2
except ModuleNotFoundError:
    pass
from plyfile import PlyData, PlyElement
from io import BytesIO
from tqdm import trange

from tools.sh_utils import RGB2SH
from tools.system_utils import mkdir_p
from tools.graphics_utils import BasicPointCloud
from tools.math_utils import normalize_pts, get_inside_normalized
from tools.general_utils import strip_symmetric, build_scaling_rotation
from tools.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from tools.denoise_pcd import remove_radius_outlier
from scene.appearance_network import AppearanceNetwork
from tools.semantic_id import BACKGROUND


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
    def __init__(self, cfg):
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.max_mem = cfg.max_mem
        
        self.use_decoupled_appearance = cfg.use_decoupled_appearance
        if self.use_decoupled_appearance:
            # appearance network and appearance embedding
            self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
            std = 1e-4
            num_embedding = len(os.listdir(os.path.join(cfg.source_path, 'images')))
            self._appearance_embeddings = nn.Parameter(torch.empty(num_embedding, 64).cuda())
            self._appearance_embeddings.data.normal_(0, std)
        
        self.enable_semantic = cfg.enable_semantic
        self._objects_dc = torch.empty(0)
        if self.enable_semantic:
            self.ch_sem_feat = cfg.ch_sem_feat
            self.num_cls = cfg.num_cls
            self.classifier = torch.nn.Conv2d(self.ch_sem_feat, self.num_cls, kernel_size=1).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scaling = self._scaling
        return self.scaling_activation(scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return self._objects_dc
    
    def get_cls(self, idx=None):
        assert self.enable_semantic, "Semantic feature is not enabled"
        feats = self.get_objects.permute(0, 2, 1)[..., None]
        if idx is not None: feats = feats[idx]
        return self.classifier(feats).view(-1, self.num_cls).argmax(-1)
    
    def logits_2_label(self, logits):
        return torch.argmax(self.logits2prob(logits), dim=-1)
    
    def logits2prob(self, logits):
        return torch.nn.functional.softmax(logits, dim=-1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    # @property
    def get_normal(self, valid=None, idx=None, refine_sign=True, is_all=False):
        '''
            rots: N, 3, 3
        '''
        normal = None
        if valid is None:
            if is_all:
                valid = torch.ones(self.get_xyz.shape[0], device='cuda', dtype=torch.bool)
            else:
                valid = self.get_inside_gaus_normalized()[0]
                normal = torch.zeros_like(self.get_xyz)
        
        _rot = self.get_rotation[valid]
        if idx is not None: _rot = _rot[idx]
        
        rots = build_rotation(_rot)
        scaling = self.get_scaling[valid]
        if idx is not None: scaling = scaling[idx]
        axis = torch.argmin(scaling, dim=-1)
        normals = rots.gather(2, axis[:, None, None].expand(-1, 3, -1)).squeeze(-1)
        
        if normal is not None:
            normal[valid] = normals
            normals = normal
        return normals
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.enable_semantic:
            # random init obj_id now
            fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0], self.ch_sem_feat), device="cuda"))
            fused_objects = fused_objects[:,:,None]
            self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args, neural_sdf_params=None):
        self.percent_dense = training_args.percent_dense
        self.large_percent_dense = None
        if hasattr(training_args, 'densify_large'):
            self.large_percent_dense = training_args.densify_large.percent_dense if \
                getattr(training_args.densify_large, 'percent_dense', 0) > 0 else None
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.use_decoupled_appearance:
            l.append({'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"})
            l.append({'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"})
        if self.enable_semantic:
            l.append({'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"})
            l.append({'params': self.classifier.parameters(), 'lr': training_args.cls_lr, "name": "classifier"})
        if neural_sdf_params is not None:
            l.append({'params': neural_sdf_params.parameters(), 'lr': training_args.sdf_lr, "name": "neural_sdf"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.enable_semantic:
            for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
                l.append('obj_dc_{}'.format(i))
        return l
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.enable_semantic:
            obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.enable_semantic:
            attributes = np.concatenate((attributes, obj_dc), axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        state_dict = {}
        if self.use_decoupled_appearance:
            state_dict["appearance_network"] = self.appearance_network.state_dict()
            state_dict["appearance_embeddings"] = self._appearance_embeddings
        if self.enable_semantic:
            state_dict["classifier"] = self.classifier.state_dict()
        if len(state_dict) > 0:
            torch.save(state_dict, os.path.join(os.path.dirname(path), 'model.pth'))

    @torch.no_grad()
    def save_inside_ply(self, path, inside=None):
        mkdir_p(os.path.dirname(path))
        
        if inside is None:
            inside = self.get_inside_gaus_normalized()[0]
        
        xyz = self._xyz[inside].detach()
        _normals = self.get_normal(inside, refine_sign=True).detach()
        normals = _normals
        
        inside = inside.cpu().numpy()
        xyz = xyz.cpu().numpy()
        normals = normals.cpu().numpy()
        
        f_dc = self._features_dc[inside].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[inside].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[inside].detach().cpu().numpy()
        scale = self._scaling[inside].detach().cpu().numpy()
        rotation = self._rotation[inside].detach().cpu().numpy()
        if self.enable_semantic:
            obj_dc = self._objects_dc[inside].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.enable_semantic:
            attributes = np.concatenate((attributes, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_visi_ply(self, path, visi):
        inside = self.get_inside_gaus_normalized()[0]
        inside = inside & visi
        
        self.save_inside_ply(path, inside)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if self.enable_semantic:
            objects_dc = np.zeros((xyz.shape[0], self.ch_sem_feat, 1))
            for idx in range(self.ch_sem_feat):
                objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.enable_semantic:
            self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree
        
        ckpt_path = os.path.join(os.path.dirname(path), 'model.pth')
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            if self.enable_semantic:
                self.classifier.load_state_dict(state_dict["classifier"])
            if self.use_decoupled_appearance:
                self.appearance_network.load_state_dict(state_dict["appearance_network"])
                self._appearance_embeddings = nn.Parameter(state_dict["appearance_embeddings"].cuda())

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network", "classifier"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network", "classifier"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.enable_semantic:
            self._objects_dc = optimizable_tensors["obj_dc"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network", "classifier"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc=None, reset=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.enable_semantic:
            d["obj_dc"] = new_objects_dc

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.enable_semantic:
            self._objects_dc = optimizable_tensors["obj_dc"]

        if reset:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        else:
            self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
            self.denom = torch.cat((self.denom, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
            self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, visi=None, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if self.large_percent_dense is not None:
            densify_pts_mask = torch.max(self.get_scaling, dim=1).values > self.large_percent_dense * scene_extent
            inside, _ = self.get_inside_gaus_normalized()
            densify_pts_mask = torch.logical_and(densify_pts_mask, inside)
            if visi is not None:
                padded_vis = torch.zeros((n_init_points), device="cuda").bool()
                padded_vis[:visi.shape[0]] = visi
                densify_pts_mask = torch.logical_and(densify_pts_mask, padded_vis)
            selected_pts_mask = torch.logical_or(selected_pts_mask, densify_pts_mask)
            
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1) if self.enable_semantic else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def get_dir_max_scaling(self, scaling, rots):
        '''
            rots: N, 3, 3
        '''
        axis = torch.argmax(scaling, dim=-1)
        max_scaling = scaling[torch.arange(scaling.shape[0]), axis]
        dirs = rots.gather(2, axis[:, None, None].expand(-1, 3, -1)).squeeze(-1)
        
        return dirs, max_scaling, axis
    
    def densify_and_split_along_maxscaling(self, grads, grad_threshold, scene_extent, visi=None, N=2, n_std=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if self.large_percent_dense is not None and (torch.cuda.memory_allocated(0) / 1024**3 < self.max_mem):
            densify_pts_mask = torch.max(self.get_scaling, dim=1).values > self.large_percent_dense * scene_extent
            inside, _ = self.get_inside_gaus_normalized()
            densify_pts_mask = torch.logical_and(densify_pts_mask, inside)
            if visi is not None:
                padded_vis = torch.zeros((n_init_points), device="cuda").bool()
                padded_vis[:visi.shape[0]] = visi
                densify_pts_mask = torch.logical_and(densify_pts_mask, padded_vis)
            selected_pts_mask = torch.logical_or(selected_pts_mask, densify_pts_mask)
            
        scaling = self.get_scaling[selected_pts_mask]
        rots = build_rotation(self._rotation[selected_pts_mask])
        dirs, max_scaling, axis = self.get_dir_max_scaling(scaling, rots)
        radii = (n_std * max_scaling / 3.)[..., None] # 3 std
        new_xyz1 = self.get_xyz[selected_pts_mask] + dirs * radii
        new_xyz2 = self.get_xyz[selected_pts_mask] - dirs * radii
        new_xyz = torch.cat((new_xyz1, new_xyz2), dim=0)
        
        new_scaling = scaling.detach().clone()
        new_scaling[torch.arange(new_scaling.shape[0]), axis] = max_scaling / (0.8*N)
        new_scaling = self.scaling_inverse_activation(new_scaling)
        new_scaling = torch.cat((new_scaling, new_scaling), dim=0)
        
        new_rotation = self._rotation[selected_pts_mask]
        new_rotation = torch.cat((new_rotation, new_rotation), dim=0)
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_dc = torch.cat((new_features_dc, new_features_dc), dim=0)
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_rest = torch.cat((new_features_rest, new_features_rest), dim=0)
        
        new_opacity = self._opacity[selected_pts_mask]
        new_opacity = torch.cat((new_opacity, new_opacity), dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1) if self.enable_semantic else None
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask] if self.enable_semantic else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, visi=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_along_maxscaling(grads, max_grad, extent, visi=visi)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        # TODO(Kevin) Emergent, change it back. This is just for testing
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def get_inside_gaus_normalized(self):
        inside, pts = get_inside_normalized(self.get_xyz, self.trans, self.scale)
        return inside, pts
    
    def normalize_pts(self, pts):
        pts = normalize_pts(pts, self.trans, self.scale)
        return pts
    
    def filter_points(self, nb_points=5, radius=0.01, std_ratio=0.01):
        inside, _ = self.get_inside_gaus_normalized()
        
        xyz = self.get_xyz[inside]
        filte_valid = remove_radius_outlier(xyz, nb_points, radius*self.extent)
        inside[inside.clone()] = filte_valid
        return inside
    
    def prune_outside(self):
        inside, _ = self.get_inside_gaus_normalized()
        self.prune_points(~inside)

    def prune_outliers(self):
        mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        valid = self.filter_points()
        mask[valid] = False
        self.prune_points(mask)

    def prune_semantics(self, cls=BACKGROUND):
        mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        mask[self.get_cls() != cls] = False
        self.prune_points(mask)
    

if __name__ == '__main__':
    model = GaussianModel(2)
    m2 = deepcopy(model)
    
