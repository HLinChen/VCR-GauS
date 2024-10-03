import os
import sys
import math
import torch
import argparse
import numpy as np
import open3d as o3d
import open3d.core as o3c

sys.path.append(os.getcwd())
from configs.config import Config
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tools.semantic_id import BACKGROUND
from tools.graphics_utils import depth2point
from tools.general_utils import set_random_seed
from tools.math_utils import get_inside_normalized
from tools.mesh_utils import GaussianExtractor, post_process_mesh


@torch.no_grad()
def tsdf_fusion(args, cfg, model, cameras, dirs, bg, outdir, mesh_name='fused_mesh.ply', max_depth=5.0):
    o3d_device = o3d.core.Device("CUDA:0")
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=args.voxel_size,
            block_resolution=16,
            block_count=60000,
            device=o3d_device)
    
    with torch.no_grad():
        for _, view in enumerate(cameras):
            
            render_pkg = render(view, model, cfg, bg, dirs=dirs)
            if args.depth_mode == 'mean':
                depth = render_pkg["depth"]
            elif args.depth_mode == 'median':
                depth = render_pkg["median_depth"]
            rgb = render_pkg["render"]
            alpha = render_pkg["alpha"]
            
            if view.gt_alpha_mask is not None:
                depth[(view.gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < args.alpha_thres)] = 0
            
            rendered_pcd_world = depth2point(depth[0], view.intr, view.world_view_transform.transpose(0, 1))[1]
            inside = get_inside_normalized(rendered_pcd_world.view(-1, 3), model.trans, model.scale)[0]
            depth.view(-1)[~inside] = 0
            
            if 'render_sem' in render_pkg:
                semantic = render_pkg["render_sem"]
                prob = model.logits2prob(semantic)
                mask = (prob[..., BACKGROUND] > args.prob_thres)[None]
                depth[mask] = 0
            
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width=view.image_width, 
                    height=view.image_height, 
                    cx = view.image_width/2,
                    cy = view.image_height/2,
                    fx = view.image_width / (2 * math.tan(view.FoVx / 2.)),
                    fy = view.image_height / (2 * math.tan(view.FoVy / 2.)))
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            
            rgb = rgb.clamp(0, 1)
            o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_color = o3d_color.to(o3d_device)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)#.to(o3d_device)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)#.to(o3d_device)
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, 1.0, max_depth)

            vbg.integrate(frustum_block_coords, o3d_depth, o3d_color, intrinsic,
                          intrinsic, extrinsic, 1.0, max_depth)
        
        mesh = vbg.extract_triangle_mesh().to_legacy()
        
        # write mesh
        o3d.io.write_triangle_mesh(os.path.join(outdir, mesh_name), mesh)

        # Clean Mesh
        if args.clean:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(outdir, mesh_name))
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_null_faces()
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=20000)
            ms.save_current_mesh(os.path.join(outdir, mesh_name))
        
        with open(os.path.join(outdir, 'voxel_size.txt'), 'w') as f:
            f.write(f'voxel_size: {args.voxel_size}')


def tsdf_cpu(args, cfg, model, cameras, dirs, bg, outdir, mesh_name='fused_mesh.ply', max_depth=5.0):
    gaussExtractor = GaussianExtractor(model, render, cfg, bg_color=bg, dirs=dirs, prob_thres=args.prob_thres, alpha_thres=args.alpha_thres)
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(cameras)
    # extract the mesh and save
    if args.unbounded:
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    else:
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=args.voxel_size, sdf_trunc=5*args.voxel_size, depth_trunc=max_depth)
    
    o3d.io.write_triangle_mesh(os.path.join(outdir, mesh_name), mesh)
    print("mesh saved at {}".format(os.path.join(outdir, mesh_name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(outdir, mesh_name), mesh_post)
    
    
    return


def main(args):
    cfg = Config(args.cfg_path)
    cfg.model.data_device = 'cpu'
    cfg.model.load_normal = False
    cfg.model.load_mask = False
    args.voxel_size = cfg.model.mesh.voxel_size if args.voxel_size == 0 else args.voxel_size
    
    set_random_seed(cfg.seed)
    
    model = GaussianModel(cfg.model)
    
    scene = Scene(cfg.model, model, load_iteration=-1, shuffle=False)
    model.trans = torch.from_numpy(scene.trans).cuda()
    model.scale = torch.from_numpy(scene.scale).cuda() * 1.1
    model.extent = scene.cameras_extent
    cameras = scene.getTrainCameras().copy()[::args.split]
    
    model.training_setup(cfg.optim)
    model.max_radii2D = torch.zeros((model.get_xyz.shape[0]), device="cuda")
    model.scale = torch.from_numpy(scene.scale).cuda()
    
    model.prune_outliers()
    
    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    print(f'Fusing into {args.mesh_name} vs: {args.voxel_size}...')
    if args.method == 'tsdf':
        dirs = scene.dirs
        max_depth = (model.scale ** 2).sum().sqrt().item()
        max_depth = args.max_depth
        tsdf_fusion(args, cfg, model, cameras, dirs, background, cfg.logdir, args.mesh_name, max_depth)
    elif args.method == 'tsdf_cpu':
        dirs = scene.dirs
        max_depth = args.max_depth
        tsdf_cpu(args, cfg, model, cameras, dirs, background, cfg.logdir, args.mesh_name, max_depth)
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='Barn')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--mesh_name', type=str, default='vcr_gaus.ply')
    parser.add_argument('--scene', type=str, default='Barn')
    parser.add_argument('--data_path', type=str, default='Barn')
    parser.add_argument('--method', type=str, default='tsdf', choices=['tsdf', 'point2mesh', 'tsdf_cpu'])
    parser.add_argument('--depth_mode', type=str, default='mean', choices=['mean', 'median'])
    parser.add_argument('--rec_method', type=str, default='poisson', choices=['nksr', 'poisson'])
    parser.add_argument('--split', type=int, default=3)
    parser.add_argument('--resolution', type=float, default=1.0)
    parser.add_argument('--detail_level', type=float, default=1.0)
    parser.add_argument('--voxel_size', type=float, default=5e-3)
    parser.add_argument('--sdf_trunc', type=float, default=0.08)
    parser.add_argument('--alpha_thres', type=float, default=0.5)
    parser.add_argument('--prob_thres', type=float, default=0.15)
    parser.add_argument('--mise_iter', type=int, default=1)
    parser.add_argument('--depth', type=int, default=9)
    parser.add_argument('--max_depth', type=float, default=6.0)
    parser.add_argument('--est_normal', action='store_true')
    parser.add_argument('--cfg_path', type=str, default='configs/config_base.yaml')
    parser.add_argument('--clean', action='store_true', help='perform a clean operation')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--num_cluster", default=1000, type=int, help='Mesh: number of connected clusters to export')
    args = parser.parse_args()
    
    main(args)