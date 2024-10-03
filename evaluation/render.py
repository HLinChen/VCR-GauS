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
import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
sys.path.append(os.getcwd())

from scene import Scene
from gaussian_renderer import render, render_fast
from gaussian_renderer import GaussianModel
from configs.config import Config
from tools.general_utils import set_random_seed
from tools.loss_utils import cos_weight


def render_set(model_path, name, iteration, views, gaussians, cfg, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    alphas = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        outs = render(view, gaussians, cfg, background)
        # outs = render_fast(view, gaussians, cfg, background)
        
        rendering = outs["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        alphas.append(outs["alpha"].detach().clone().view(-1).cpu())
        
        if False:
            normal_map = outs["normal"].detach().clone()
            normal_gt = view.normal.cuda()
            cos = cos_weight(normal_gt, normal_map, cfg.optim.exp_t, cfg.optim.cos_thr)
            torchvision.utils.save_image(cos, os.path.join(render_path, '{0:05d}_cosine'.format(idx) + ".png"))
    
    # alphas = torch.cat(alphas, dim=0)
    # print("Alpha min: {}, max: {}".format(alphas.min(), alphas.max()))
    # print("Alpha mean: {}, std: {}".format(alphas.mean(), alphas.std()))
    # print("Alpha median: {}".format(alphas.median()))


def render_sets(cfg, iteration : int, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(cfg.model)
        scene = Scene(cfg.model, gaussians, load_iteration=iteration, shuffle=False)
        # gaussians.extent = scene.cameras_extent

        bg_color = [1,1,1] if cfg.model.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(cfg.model.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, cfg, background)

        if not skip_test:
            render_set(cfg.model.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, cfg, background)
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='configs/config_base.yaml')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()
    
    cfg = Config(args.cfg_path)
    cfg.model.data_device = 'cuda'
    cfg.model.load_normal = False
    cfg.model.load_mask = False
    
    set_random_seed(cfg.seed)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(cfg, args.iteration, args.skip_train, args.skip_test)