import os
import json
import uuid
import math
import wandb
import imageio
import numpy as np
from torch import nn
from tqdm import tqdm
import math
from random import randint
import torch.nn.functional as F
from argparse import Namespace
from pytorch3d.ops import knn_points
from torchmetrics import JaccardIndex

import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from tools.loss_utils import l1_loss, ssim, cos_weight, entropy_loss, monosdf_normal_loss, ScaleAndShiftInvariantLoss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tools.image_utils import psnr
from configs.config import Config
from tools.visualization import wandb_image, preprocess_image, wandb_sem
from tools.prune import prune_list, calculate_v_imp_score, get_visi_list
from tools.loss_utils import compute_normal_loss, L1_loss_appearance, normal2curv
from tools.camera_utils import bb_camera
from tools.general_utils import safe_state, set_random_seed
from scene.cameras import SampleCam
from tools.normal_utils import get_normal_sign, get_edge_aware_distortion_map
# from process_data.extract_mask import text_label_dict

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        cfg.model.model_path = cfg.logdir
        self.sphere = getattr(cfg.model, 'sphere', False)
        cfg.model.load_normal = cfg.optim.loss_weight.mono_normal > 0 \
                                    or cfg.optim.loss_weight.depth_normal > 0
        cfg.model.load_depth = cfg.optim.loss_weight.mono_depth > 0
        self.enable_semantic = getattr(cfg.optim.loss_weight, 'semantic', 0) > 0
        cfg.model.enable_semantic = self.enable_semantic
        cfg.model.load_mask = self.enable_semantic or cfg.model.load_mask
        cfg.print_config()
        safe_state(cfg.silent)
        
        self.setup_model(cfg.model)
        self.setup_dataset(cfg.model)
        self.setup_optimizer(cfg.optim)
        self.init_attributes()
        self.init_losses()
        
        # Start GUI server, configure and run training
        if cfg.port > 0:
            network_gui.init(cfg.ip, cfg.port)
        torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    def setup_model(self, cfg):
        self.model = GaussianModel(cfg)

    def setup_dataset(self, cfg):
        os.makedirs(cfg.model_path, exist_ok = True)
        self.scene = Scene(cfg, self.model)
        self.model.trans = torch.from_numpy(self.scene.trans).cuda()
        self.model.scale = torch.from_numpy(self.scene.scale).cuda()
        self.model.extent = self.scene.cameras_extent
    
    def init_writer(self, cfg):
        if not cfg.model.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            cfg.model.model_path = os.path.join("./output/", unique_str[0:10])
            
        # Set up output folder
        print("Output folder: {}".format(cfg.model.model_path))
        os.makedirs(cfg.model.model_path, exist_ok = True)
        with open(os.path.join(cfg.model.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(cfg))))

        # Create Tensorboard writer
        if TENSORBOARD_FOUND:
            self.writer = SummaryWriter(cfg.model.model_path)
        else:
            print("Tensorboard not available: not logging progress")
    
    def init_wandb(self, cfg, wandb_id=None, project="", run_name=None, mode="online", resume="allow", use_group=False):
        r"""Initialize Weights & Biases (wandb) logger.

        Args:
            cfg (obj): Global configuration.
            wandb_id (str): A unique ID for this run, used for resuming.
            project (str): The name of the project where you're sending the new run.
                If the project is not specified, the run is put in an "Uncategorized" project.
            run_name (str): name for each wandb run (useful for logging changes)
            mode (str): online/offline/disabled
        """
        print('Initialize wandb')
        if not wandb_id:
            wandb_path = os.path.join(cfg.logdir, "wandb_id.txt")
            if os.path.exists(wandb_path):
                with open(wandb_path, "r") as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(wandb_path, "w") as f:
                    f.write(wandb_id)
        if use_group:
            group, name = cfg.logdir.split("/")[-2:]
        else:
            group, name = None, os.path.basename(cfg.logdir)

        if run_name is not None:
            name = run_name

        wandb.init(id=wandb_id,
                    project=project,
                    config=cfg,
                    group=group,
                    name=name,
                    dir=cfg.logdir,
                    resume=resume,
                    settings=wandb.Settings(start_method="fork"),
                    mode=mode)
        wandb.config.update({'dataset': cfg.data.name})

    def init_losses(self):
        r"""Initialize loss functions. All loss names have weights. Some have criterion modules."""
        self.losses = dict()
        
        self.weights = {key: value for key, value in self.cfg.optim.loss_weight.items() if value}
        
        if 'mono_depth' in self.weights:
            self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    
    def setup_optimizer(self, cfg):
        self.model.training_setup(cfg)
    
    def init_attributes(self):
        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        
        self.current_iteration = 0
        self.max_iters = self.cfg.optim.iterations
        self.saving_iterations = self.cfg.train.save_iterations
        self.testing_iterations = self.cfg.train.test_iterations
        self.checkpoint_iterations = self.cfg.train.checkpoint_iterations
        
        self.debug_from = self.cfg.train.debug_from
        self.checkpoint = self.cfg.train.start_checkpoint
        self.star_ft_iter = None
        
        self.visi_list = None
        
        self.first_iter = 0
        if self.checkpoint:
            (model_params, self.first_iter) = torch.load(self.checkpoint)
            self.model.restore(model_params, self.cfg.optim)
            
        bg_color = [1, 1, 1] if self.cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.writer = None
        
        with open(os.path.join(self.cfg.model.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(self.cfg))))
        
        self.vis_path = os.path.join(self.cfg.logdir, "vis")
        self.vis_color_path = os.path.join(self.vis_path, "color")
        self.vis_depth_path = os.path.join(self.vis_path, "depth")
        self.vis_normal_path = os.path.join(self.vis_path, "normal")
        self.vis_dnormal_path = os.path.join(self.vis_path, "dnormal")
        self.vis_cos_path = os.path.join(self.vis_path, "cos")
        
        for mode in ['train', 'test']:
            os.makedirs(os.path.join(self.vis_color_path, mode), exist_ok=True)
            os.makedirs(os.path.join(self.vis_depth_path, mode), exist_ok=True)
            os.makedirs(os.path.join(self.vis_normal_path, mode), exist_ok=True)
            os.makedirs(os.path.join(self.vis_dnormal_path, mode), exist_ok=True)
            os.makedirs(os.path.join(self.vis_cos_path, mode), exist_ok=True)
        
        
        if self.enable_semantic:
            self.calc_miou = JaccardIndex(num_classes=self.model.num_cls, task='multiclass').cuda()
    
    def train(self):
        progress_bar = tqdm(range(self.first_iter, self.max_iters), desc="Training progress")
        self.current_iteration += self.first_iter
        self.first_iter += 1
        for iteration in range(self.first_iter, self.max_iters  + 1):
            self.current_iteration += 1
            
            self.start_of_iteration()
            
            output = self.train_step(mode='train')
            
            self.end_of_iteration(output, render, progress_bar)
    
    def get_center_scale(self):
        meta_fname = f"{self.cfg.model.source_path}/meta.json"
        with open(meta_fname) as file:
            meta = json.load(file)
        # center scene
        trans = np.array(meta["trans"], dtype=np.float32)
        trans = torch.from_numpy(trans.astype(np.float32)).to("cuda")
        self.model.trans = torch.nn.parameter.Parameter(trans, requires_grad=False)
        # scale scene
        scale = np.array(meta["scale"], dtype=np.float32)
        scale = torch.from_numpy(scale.astype(np.float32)).to("cuda")
        self.model.scale = torch.nn.parameter.Parameter(scale, requires_grad=False)

    def model_forward(self, data, mode):
        render_pkg = render(data['viewpoint_cam'], self.model, self.cfg, data.pop('bg'), dirs=self.scene.dirs)
        data.update(render_pkg)
        self._compute_loss(data, mode)
        loss = self._get_total_loss()
        
        return loss
    
    def _compute_loss(self, data, mode=None):
        if mode == 'train':
            gt_image = data['viewpoint_cam'].original_image.cuda()
            self.losses['l1'] = l1_loss(data['render'], gt_image) if not self.cfg.model.use_decoupled_appearance \
                else L1_loss_appearance(data['render'], gt_image, self.model, data['viewpoint_cam'].idx)
            self.losses['ssim'] = 1.0 - ssim(data['render'], gt_image)
            
            if 'l1_scale' in self.weights or 'entropy' in self.weights or 'proj' in self.weights or 'repul' in self.weights:
                mask, _ = self.model.get_inside_gaus_normalized()
            
            if 'l1_scale' in self.weights and not self.sphere:
                scaling = self.model.get_scaling[mask].min(-1)[0]
                self.losses['l1_scale'] = l1_loss(scaling, torch.zeros_like(scaling))
            
            if 'entropy' in self.weights:
                opacity = self.model.get_opacity[mask]
                self.losses['entropy'] = entropy_loss(opacity)
            
            if 'mono_depth' in self.weights:
                render_depth = data['depth']
                gt_depth = data['viewpoint_cam'].depth.cuda().float()
                mask = None
                if self.cfg.model.load_mask:
                    mask = data['viewpoint_cam'].mask
                
                mask = render_depth > 0
                self.losses['mono_depth'] = self.depth_loss(render_depth, gt_depth, mask)
            
            if 'mono_normal' in self.weights and self.current_iteration > self.cfg.optim.normal_from_iter:
                render_normal = data['normal']
                gt_normal = data['viewpoint_cam'].normal.cuda()
                self.losses['mono_normal'] = monosdf_normal_loss(render_normal, gt_normal)
            
            if 'depth_normal' in self.weights and self.current_iteration > self.cfg.optim.dnormal_from_iter:
                est_normal = data['est_normal']
                gt_normal = data['viewpoint_cam'].normal.cuda()
                render_normal = data['normal'].detach()
                mask = data['mask']
                
                with torch.no_grad():
                    weights = cos_weight(render_normal, gt_normal, self.cfg.optim.exp_t)
                
                if mask.sum() != 0:
                    est_normal, gt_normal = est_normal[mask], gt_normal[mask]
                    render_normal = render_normal[mask]
                    weights = weights[mask]
                    self.losses['depth_normal'] = monosdf_normal_loss(est_normal, gt_normal, weights)
                else: self.losses['depth_normal'] = 0
                
                if 'curv' in self.weights and self.current_iteration > self.cfg.optim.curv_from_iter:
                    est_normal = data['est_normal']         # h, w, 3
                    mask = data['mask'][..., None].clone()          # h, w, 1
                    mask = mask.float()
                    curv = normal2curv(est_normal, mask)
                    self.losses['curv'] = l1_loss(curv, 0)
                
            if 'consistent_normal' in self.weights and self.current_iteration > self.cfg.optim.consistent_normal_from_iter:
                est_normal = data['est_normal']
                render_normal = data['normal']
                mask = data['mask']
                self.losses['consistent_normal'] = monosdf_normal_loss(est_normal, render_normal)
                
            if 'distortion' in self.weights and self.current_iteration > self.cfg.optim.close_depth_from_iter:
                distortion_map = data['distortion']
                distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
                self.losses['distortion'] = distortion_map.mean()
            
            if 'depth_var' in self.weights and self.current_iteration > self.cfg.optim.close_depth_from_iter:
                depth_var = data['depth_var']
                depth_var = get_edge_aware_distortion_map(gt_image, depth_var)
                self.losses['depth_var'] = depth_var.mean()
                
            if 'semantic' in self.weights:
                sem_logits = data['render_sem']
                sem_trg = data['viewpoint_cam'].mask.view(-1)
                self.losses['semantic'] = F.cross_entropy(sem_logits.view(-1, self.model.num_cls), sem_trg) / torch.log(torch.tensor(self.model.num_cls)) # normalize to (0,1)
    
    def _get_total_loss(self):
        r"""Return the total loss to be backpropagated.
        """
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        
        # Iterates over all possible losses.
        for loss_name in self.weights:
            if loss_name in self.losses:
                # Multiply it with the corresponding weight and add it to the total loss.
                total_loss += self.losses[loss_name] * self.weights[loss_name]
        self.losses['total'] = total_loss  # logging purpose
        return total_loss
    
    def train_step(self, mode='train'):
        data = dict()
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        data['viewpoint_cam'] = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

        # Render
        if (self.current_iteration - 1) == self.debug_from:
            self.cfg.pipline.debug = True

        data['bg'] = torch.rand((3), device="cuda") if self.cfg.optim.random_background else self.background

        loss = self.model_forward(data, mode)
        
        loss.backward()
        viewspace_point_tensor, visibility_filter, radii = data.pop("viewspace_points"), data.pop("visibility_filter"), data.pop("radii")

        with torch.no_grad():
            # Densification
            if self.current_iteration < self.cfg.optim.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.model.max_radii2D[visibility_filter] = torch.max(self.model.max_radii2D[visibility_filter], radii[visibility_filter])
                viewspace_point_tensor_densify = data["viewspace_points_densify"]
                self.model.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)
                # self.model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.current_iteration > self.cfg.optim.densify_from_iter \
                    and hasattr(self.cfg.optim, 'densify_large'):
                        
                    if 'countlist' in data:
                        visi_list_each = data['countlist']
                        self.visi_list = visi_list_each if self.visi_list is None else self.visi_list + visi_list_each

                if self.current_iteration > self.cfg.optim.densify_from_iter and self.current_iteration % self.cfg.optim.densification_interval == 0:
                    size_threshold = 20 if self.current_iteration > self.cfg.optim.opacity_reset_interval else None
                    visi = None
                    
                    if getattr(self.cfg.optim, 'densify_large', False) and self.cfg.optim.densify_large.sample_cams.num > 0 \
                        and getattr(self.cfg.optim.densify_large, 'percent_dense', 0):
                        visi = self.get_visi_mask_acc(self.cfg.optim.densify_large.sample_cams.num,
                                                self.cfg.optim.densify_large.sample_cams.up,
                                                self.cfg.optim.densify_large.sample_cams.around,
                                                sample_mode='random')
                        if self.visi_list is not None:
                            visi = visi & self.visi_list > 0
                    self.model.densify_and_prune(self.cfg.optim.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold, visi)
                    self.visi_list = None
                    
                if self.current_iteration % self.cfg.optim.opacity_reset_interval == 0 or \
                    (self.cfg.model.white_background and self.current_iteration == self.cfg.optim.densify_from_iter):
                    self.model.reset_opacity()

            if self.current_iteration in self.cfg.optim.prune.iterations:
                # TODO Add prunning types
                n = int(len(self.scene.getFullCameras()) * 1.2)
                viewpoint_stack = self.scene.getFullCameras().copy()
                gaussian_list, imp_list = prune_list(self.model, viewpoint_stack, self.cfg.pipline, self.background)
                i = self.cfg.optim.prune.iterations.index(self.current_iteration)
                v_list = calculate_v_imp_score(self.model, imp_list, self.cfg.optim.prune.v_pow)
                self.model.prune_gaussians(
                    (self.cfg.optim.prune.decay**i) * self.cfg.optim.prune.percent, v_list
                )
            
            
            # Optimizer step
            self.model.optimizer.step()
            self.model.optimizer.zero_grad(set_to_none = True)
        
        return data

    def start_of_iteration(self):
        self.iter_start.record()
        
        # train or fine-tune
        iter = self.current_iteration if self.star_ft_iter is None \
            else self.current_iteration - self.star_ft_iter
        self.model.update_learning_rate(iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.current_iteration % 1000 == 0:
            self.model.oneupSHdegree()
        
    def end_of_iteration(self, output, render, progress_bar):
        self.iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * self.losses['total'].item() + 0.6 * self.ema_loss_for_log
            if self.current_iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if self.current_iteration == self.max_iters :
                progress_bar.close()

            # Log and save
            if self.writer:
                self.log_writer(output, mode="train")
            else:
                output.update(self.test(render))
                self.log_wandb_scalars(output, mode="train")
            
            if (self.current_iteration in self.saving_iterations) or (self.current_iteration == self.max_iters):
                self.save_gaussians()

            if (self.current_iteration in self.checkpoint_iterations) or (self.current_iteration == self.max_iters):
                print("\n[ITER {}] Saving Checkpoint".format(self.current_iteration))
                torch.save((self.model.capture(), self.current_iteration), self.scene.model_path + "/chkpnt" + str(self.current_iteration) + ".pth")
                
                if len(self.cfg.optim.prune.iterations) > 0 and self.current_iteration == self.max_iters:
                    viewpoint_stack = self.scene.getFullCameras().copy()
                    gaussian_list, imp_list = prune_list(self.model, viewpoint_stack, self.cfg.pipline, self.background)
                    v_list = calculate_v_imp_score(self.model, imp_list, self.cfg.optim.prune.v_pow)
                    np.savez(os.path.join(self.scene.model_path, "imp_score"), v_list.cpu().detach().numpy())
    
    def log_wandb_scalars(self, output, mode=None):
        scalars = dict()
        if mode == "train":
            for param_group in self.model.optimizer.param_groups:
                scalars.update({"optim/lr_{}".format(param_group["name"]): param_group['lr']})
                
        scalars.update({"time/iteration": self.iter_start.elapsed_time(self.iter_end)})
        scalars.update({f"loss/{mode}_{key}": value for key, value in self.losses.items()})
        scalars.update(iteration=self.current_iteration)
        
        scalars.update({k: v for k, v in output.items() if isinstance(v, (int, float))})
        
        wandb.log(scalars, step=self.current_iteration)

    def log_wandb_images(self, data, mode=None):
        image = torch.cat([data["rgb_map"], data["image"]], dim=1)
        depth = data["depth_map"]
        inv_depth = depth.max() - depth
        images = {f'vis/{mode}': wandb_image(image),
                  f'vis/{mode}_depth': wandb_image(depth, from_range=(depth.min(), depth.max())),
                  f'vis/{mode}_inv_depth': wandb_image(inv_depth, from_range=(inv_depth.min(), inv_depth.max()))}
        if 'depth_var' in data:
            depth_var = data['depth_var']
            images.update({f'vis/{mode}_depth_var': wandb_image(depth_var, from_range=(depth_var.min(), depth_var.max()))})
        if 'depth' in data:
            depth = data["depth"].detach().clone()
            images.update({f'vis/{mode}_depth_gt': wandb_image(depth, from_range=(depth.min(), depth.max()))})
        if 'mask' in data:
            mask = data['mask'].detach().clone().float()
            images.update({f'vis/{mode}_mask': wandb_image(mask)})
        if 'normal_map' in data:
            normal_map = data["normal_map"]
            images.update({f'vis/{mode}_normal': wandb_image(normal_map.permute(2, 0, 1), from_range=(-1, 1))})
            if 'normal' in data:
                normal = data["normal"].detach().clone()
                images.update({f'vis/{mode}_normal_gt': wandb_image(normal.permute(2, 0, 1), from_range=(-1, 1))})
                cos = cos_weight(normal.cuda(), normal_map, self.cfg.optim.exp_t)
                images.update({f'vis/{mode}_normal_cos': wandb_image(cos, from_range=(0, 1))})
            if 'est_normal' in data:
                est_normal = data["est_normal"].permute(2, 0, 1).detach().clone()
                images.update({f'vis/{mode}_est_normal': wandb_image(est_normal, from_range=(-1, 1))})
            if 'transformed_est_normal' in data:
                transformed_est_normal = data["transformed_est_normal"].permute(2, 0, 1).detach().clone()
                images.update({f'vis/{mode}_trans_est_normal': wandb_image(transformed_est_normal, from_range=(-1, 1))})
        if 'sem' in data:
            sem = data['sem']
            images.update({f'vis/{mode}_sem': wandb_sem(sem)})
        if 'distortion' in data:
            distortion = data['distortion']
            images.update({f'vis/{mode}_distortion': wandb_image(distortion, from_range=(distortion.min(), distortion.max()))})
        if 'depth_var' in data:
            depth_var = data['depth_var']
            images.update({f'vis/{mode}_depth_var': wandb_image(depth_var, from_range=(depth_var.min(), depth_var.max()))})
        if 'trans_image' in data:
            trans_image = data['trans_image']
            images.update({f'vis/{mode}_trans': wandb_image(trans_image)})
        wandb.log(images, step=self.current_iteration)
    
    def log_hist(self, tensor, name, num_bin=10):
        counts, bins = np.histogram(tensor, bins=num_bin)
        density = counts / counts.sum()
        plt.stairs(density, bins)
        plt.title('Histogram {}'.format(name))
        wandb.log({f'statistic/{name}': wandb.Image(plt)}, step=self.current_iteration)
        plt.close()
    
    @torch.no_grad()
    def test(self, renderFunc):
        output = dict()
        # Report test and samples of training set
        if (self.current_iteration in self.testing_iterations) or (self.current_iteration == self.max_iters):
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : self.scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : self.scene.getTrainCameras()})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        out = renderFunc(viewpoint, self.model, self.cfg, self.background, dirs=self.scene.dirs)
                        image = torch.clamp(out["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if config['name'] == 'train' and self.cfg.model.use_decoupled_appearance:
                            trans_image = L1_loss_appearance(image, gt_image, self.model, viewpoint.idx, return_transformed_image=True)
                            
                        depth = out["depth"]
                        normal = out["normal"] if "normal" in out else None
                        est_normal = out["est_normal"] if "est_normal" in out else None
                        if 'render_sem' in out:
                            pred = self.model.logits_2_label(out['render_sem'])
                            sem_mask = viewpoint.mask.cuda()
                            self.calc_miou.update(pred, sem_mask)
                        if viewpoint.image_name == self.scene.first_name:
                            data = {"image": gt_image, "rgb_map": image, "depth_map": depth}
                            if config['name'] == 'train' and self.cfg.model.use_decoupled_appearance:
                                data['trans_image'] = trans_image
                            if 'mask' in out: data['mask'] = out['mask']
                            if viewpoint.depth is not None: data['depth'] = viewpoint.depth
                            if 'depth_var' in out: data['depth_var'] = out['depth_var']
                            if 'distortion' in out: data['distortion'] = out['distortion']
                            if normal is not None:
                                data["normal_map"] = normal
                                if viewpoint.normal is not None: data['normal'] = viewpoint.normal
                                if est_normal is not None:
                                    data['est_normal'] = est_normal
                            if 'render_sem' in out:
                                pred = self.model.logits_2_label(out['render_sem']).to(torch.uint8)
                                data['sem'] = torch.cat([pred, sem_mask], dim=0)
                                
                            self.log_wandb_images(data, mode=config['name'])
                        
                        if False:
                            data = {"image": gt_image, "rgb_map": image, "depth_map": depth}
                            if 'mask' in out: data['mask'] = out['mask']
                            if viewpoint.depth is not None: data['depth'] = viewpoint.depth
                            if 'depth_var' in out: data['depth_var'] = out['depth_var']
                            if normal is not None:
                                data["normal_map"] = normal
                                if viewpoint.normal is not None: data['normal'] = viewpoint.normal
                                if est_normal is not None: data['est_normal'] = est_normal
                            cos = cos_weight(normal.cuda(), normal, self.cfg.optim.exp_t)
                            data['normal_cos'] = cos
                            self.save_vis(data, viewpoint.image_name, mode=config['name'])
                        
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    
                    if self.enable_semantic:
                        miou = self.calc_miou.compute()
                        self.calc_miou.reset()
                    
                    output.update({
                        f'statistic/{config["name"]}_PSNR': psnr_test.item(),
                        f'loss/{config["name"]}_l1': l1_test.item(),
                    })
                    if self.enable_semantic:
                        output[f'statistic/{config["name"]}_mIoU'] = miou.item()
            
            output.update({
                'statistic/total_points': self.scene.gaussians.get_xyz.shape[0],
            })
            
            self.log_hist(self.model.get_opacity.cpu().numpy(), "opacity")
            
            torch.cuda.empty_cache()
        
        return output

    def finalize(self):
        # Finish the W&B logger.
        wandb.finish()

    def log_writer(self, mode=None):
        if self.writer:
            for key, value in self.losses.items():
                self.writer.add_scalar(f"loss/{mode}_{key}", value, global_step=self.current_iteration)
    
    def save_vis(self, data, name, mode='train'):
        image = torch.clamp(data["rgb_map"], 0.0, 1.0).detach().cpu()
        image = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
        imageio.imsave(os.path.join(self.vis_color_path, mode, f"{name}.png"), image)
        
        normal = preprocess_image(data["normal_map"].permute(2, 0, 1), from_range=(-1, 1))
        normal.save(os.path.join(self.vis_normal_path, mode, f"{name}.png"))
        
        if False:
            normal_gt = preprocess_image(data["normal"].permute(2, 0, 1), from_range=(-1, 1))
            gt_normal_path = os.path.join(self.vis_normal_path+'_gt', mode)
            if not os.path.exists(gt_normal_path):
                os.makedirs(gt_normal_path, exist_ok=True)
            normal_gt.save(os.path.join(gt_normal_path, f"{name}.png"))
        
        dnormal = preprocess_image(data["est_normal"].permute(2, 0, 1), from_range=(-1, 1))
        dnormal.save(os.path.join(self.vis_dnormal_path, mode, f"{name}.png"))
        
        cos = preprocess_image(data["normal_cos"], from_range=(0, 1))
        cos.save(os.path.join(self.vis_cos_path, mode, f"{name}.png"))
        
        return

    def sample_cameras(self, n, up=False, around=True, look_mode='target', sample_mode='grid', bidirect=True): # direction target
        cam_height = None
        w2cs = bb_camera(n, self.model.trans, self.model.scale, cam_height, up=up, around=around, \
            look_mode=look_mode, sample_mode=sample_mode, bidirect=bidirect)
        FoVx = FoVy = 2.5
        width = height = 1500
        cams = []
        
        for i in range(w2cs.shape[0]):
            w2c = w2cs[i]
            cam = SampleCam(w2c, width, height, FoVx, FoVy)
            cams.append(cam)
        
        return cams
    
    @torch.no_grad()
    def get_visi_mask(self, n=500, up=False, around=True, denoise_after=False, \
        denoise_before=True, nb_points=10, viewpoint_stack=None, sample_mode='grid', cat_cams=False): # direction target
        if viewpoint_stack is None:
            if self.cfg.optim.densify_large.sample_cams.random:
                viewpoint_stack = self.sample_cameras(n, up, around, sample_mode=sample_mode)
                if cat_cams:
                    viewpoint_stack += self.scene.getTrainCameras().copy()
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
        
        model = deepcopy(self.model)
        
        if denoise_before:
            mask = torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            valid = model.filter_points()
            mask[valid] = False
        
            model.prune_points(mask)
        else:
            mask = torch.zeros(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        xyz = model.get_xyz[None]
        dist2 = knn_points(xyz, xyz, K=nb_points+1, return_sorted=True).dists # 1, N, K
        dist2 = dist2[0, :, 1:]
        dist2 = torch.clamp_min(dist2, 0.0000001)
        dist = (torch.sqrt(dist2)).mean(-1)
        scaling = dist
        
        scales = torch.log(scaling)[...,None].repeat(1, 3)
        
        idx = torch.argmin(model.get_scaling, dim=-1)
        scales[torch.arange(scales.shape[0]), idx] = math.log(1e-7)
        model._scaling = nn.Parameter(scales.requires_grad_(True))
        
        out = get_visi_list(model, viewpoint_stack, self.cfg.pipline, self.background)
        
        visi = out['visi']
        
        valid = ~mask
        if denoise_after:
            model.prune_points(~visi)
            filted = model.filter_points()
            visi[visi.clone()] = filted
            
        valid[~mask] = visi
        
        del model
        
        return valid

    @torch.no_grad()
    def get_visi_mask_acc(self, n=500, up=False, around=True, sample_mode='grid', viewpoint_stack=None):
        if viewpoint_stack is None:
            if self.cfg.optim.densify_large.sample_cams.random:
                viewpoint_stack = self.sample_cameras(n, up, around, sample_mode=sample_mode)
            else:
                fullcam = self.scene.getTrainCameras().copy()
                idx = torch.randint(0, len(fullcam), (n,))
                viewpoint_stack = [fullcam[i] for i in idx]
            
        out = get_visi_list(self.model, viewpoint_stack, self.cfg.pipline, self.background)
        visi = out['visi']
        inside = self.model.get_inside_gaus_normalized()[0]
        valid = visi & inside
        
        return valid

    @torch.no_grad()
    def save_gaussians(self):
        print("\n[ITER {}] Saving Gaussians".format(self.current_iteration))
        
        surfmask = None
        visi = None
        self.scene.save(self.current_iteration, visi=visi, surf=surfmask, save_splat=self.cfg.train.save_splat)
    

if __name__ == "__main__":
    from configs.config import Config
    import sys
    sys.path.append(os.getcwd())
    
    cfg_path = 'projects/gaussain_splatting/configs/base.yaml'
    
    cfg = Config(cfg_path)
    
    trainer = Trainer(cfg)
    
    trainer.get_center_scale()
    
    for thr in np.linspace(0.9, 1., 11):
        trainer.save_pts_thr(thr)
    
