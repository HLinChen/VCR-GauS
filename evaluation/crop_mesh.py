import os
import json
import plyfile
import argparse
# import open3d as o3d
import numpy as np
# from tqdm import tqdm
import trimesh
from sklearn.cluster import DBSCAN


def align_gt_with_cam(pts, trans):
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def main(args):
    assert os.path.exists(args.ply_path), f"PLY file {args.ply_path} does not exist."
    gt_trans = np.loadtxt(args.align_path)
    
    mesh_rec = trimesh.load(args.ply_path, process=False)
    mesh_gt = trimesh.load(args.gt_path, process=False)
    
    mesh_gt.vertices = align_gt_with_cam(mesh_gt.vertices, gt_trans)
    
    to_align, _ = trimesh.bounds.oriented_bounds(mesh_gt)
    mesh_gt.vertices = (to_align[:3, :3] @ mesh_gt.vertices.T + to_align[:3, 3:]).T
    mesh_rec.vertices = (to_align[:3, :3] @ mesh_rec.vertices.T + to_align[:3, 3:]).T
    
    min_points = mesh_gt.vertices.min(axis=0)
    max_points = mesh_gt.vertices.max(axis=0)

    mask_min = (mesh_rec.vertices - min_points[None]) > 0
    mask_max = (mesh_rec.vertices - max_points[None]) < 0

    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[mesh_rec.faces].all(axis=1)

    mesh_rec.update_vertices(mask)
    mesh_rec.update_faces(face_mask)
    
    mesh_rec.vertices = (to_align[:3, :3].T @ mesh_rec.vertices.T - to_align[:3, :3].T @ to_align[:3, 3:]).T
    mesh_gt.vertices = (to_align[:3, :3].T @ mesh_gt.vertices.T - to_align[:3, :3].T @ to_align[:3, 3:]).T
    
    # save mesh_rec and mesh_rec in args.out_path
    mesh_rec.export(args.out_path)
    
    # downsample mesh_gt
    
    idx = np.random.choice(np.arange(len(mesh_gt.vertices)), 5000000)
    mesh_gt.vertices = mesh_gt.vertices[idx]
    mesh_gt.colors = mesh_gt.colors[idx]
    
    mesh_gt.export(args.gt_path.replace('.ply', '_trans.ply'))
    
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_path",
        type=str,
        default='/your/path//Barn_GT.ply',
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--align_path",
        type=str,
        default='/your/path//Barn_trans.txt',
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default='/your/path//Barn_lowres.ply',
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default='Barn',
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default='/your/path//Barn_lowres_crop.ply',
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    args = parser.parse_args()
    
    main(args)