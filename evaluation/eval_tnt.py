import os
import trimesh
import argparse
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics


def main(args):
    assert os.path.exists(args.ply_path), f"PLY file {args.ply_path} does not exist."
    
    mesh_rec = trimesh.load(args.ply_path, process=False)
    mesh_gt = trimesh.load(args.gt_path, process=False)
    
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
    
    metrics = evaluate(mesh_rec, mesh_gt)
    
    metrics_path = os.path.join(os.path.dirname(args.ply_path), 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
    
    print('Scene: {} F-score: {}'.format(args.scene, metrics['F-score']))
    
    mesh_rec.vertices = (to_align[:3, :3].T @ mesh_rec.vertices.T - to_align[:3, :3].T @ to_align[:3, 3:]).T
    mesh_rec.export(args.ply_path.replace('.ply', '_crop.ply'))
    
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
    args = parser.parse_args()
    
    main(args)

