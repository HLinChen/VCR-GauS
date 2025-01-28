'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import numpy as np
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
import json
import trimesh

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
sys.path.append(dir_path.__str__())

from submodules.colmap.scripts.python.read_write_model import read_model, qvec2rotmat  # NOQA


def find_closest_point(p1, d1, p2, d2):
    # Calculate the direction vectors of the lines
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Create the coefficient matrix A and the constant vector b
    A = np.vstack((d1_norm, -d2_norm)).T
    b = p2 - p1

    # Solve the linear system to find the parameters t1 and t2
    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the closest point on each line
    closest_point1 = p1 + d1_norm * t1
    closest_point2 = p2 + d2_norm * t2

    # Calculate the average of the two closest points
    closest_point = 0.5 * (closest_point1 + closest_point2)

    return closest_point


def bound_by_pose(images):
    poses = []
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)

    center = np.array([0.0, 0.0, 0.0])
    for f in poses:
        src_frame = f[0:3, :]
        for g in poses:
            tgt_frame = g[0:3, :]
            p = find_closest_point(src_frame[:, 3], src_frame[:, 2], tgt_frame[:, 3], tgt_frame[:, 2])
            center += p
    center /= len(poses) ** 2

    radius = 0.0
    for f in poses:
        radius += np.linalg.norm(f[0:3, 3])
    radius /= len(poses)
    bounding_box = [
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    ]
    return center, radius, bounding_box


def bound_by_points(points3D):
    if not isinstance(points3D, np.ndarray):
        xyzs = np.stack([point.xyz for point in points3D.values()])
    else:
        xyzs = points3D
    center = xyzs.mean(axis=0)
    std = xyzs.std(axis=0)
    # radius = float(std.max() * 2)  # use 2*std to define the region, equivalent to 95% percentile
    radius = np.abs(xyzs).max(0) * 1.1
    bounding_box = [
        [center[0] - std[0] * 3, center[0] + std[0] * 3],
        [center[1] - std[1] * 3, center[1] + std[1] * 3],
        [center[2] - std[2] * 3, center[2] + std[2] * 3],
    ]
    return center, radius, bounding_box


def compute_oriented_bound(pts):
    to_align, _ = trimesh.bounds.oriented_bounds(pts)
    
    scale = (np.abs((to_align[:3, :3] @ pts.vertices.T + to_align[:3, 3:]).T).max(0) * 1.2).tolist()
    
    return to_align.tolist(), scale


def split_data(names, split=10):
    split_dict = {'train': [], 'test': []}
    names = sorted(names)
    
    for i, name in enumerate(names):
        if i % split == 0:
            split_dict['test'].append(name)
        else:
            split_dict['train'].append(name)
    
    split_dict['train'] = sorted(split_dict['train'])
    split_dict['test'] = sorted(split_dict['test'])
    return split_dict


def get_split_dict(scene_path):
    split_dict = None
    
    if os.path.exists(os.path.join(scene_path, 'train_test_lists.json')):
        image_names = os.listdir(os.path.join(scene_path, "images"))
        image_names = sorted(['{:06}'.format(int(i.split(".")[0])) for i in image_names])
        
        with open(os.path.join(scene_path, 'train_test_lists.json'), 'r') as fp:
            split_dict = json.load(fp)
            
        test_split = sorted([i.split(".")[0] for i in split_dict['test']])
        train_split = [i for i in image_names if i not in test_split]
        
        assert len(train_split) + len(test_split) == len(image_names), "train and test split do not cover all images"
        
        split_dict = {
            'train': train_split,
            'test': test_split,
        }
    
    return split_dict


def check_concentric(images, ang_tol=np.pi / 6.0, radii_tol=0.5, pose_tol=0.5):
    look_at = []
    cam_loc = []
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        cam_loc.append(c2w[:3, -1])
        look_at.append(c2w[:3, 2])
    look_at = np.stack(look_at)
    look_at = look_at / np.linalg.norm(look_at, axis=1, keepdims=True)
    cam_loc = np.stack(cam_loc)
    num_images = cam_loc.shape[0]

    center = cam_loc.mean(axis=0)
    vec = center - cam_loc
    radii = np.linalg.norm(vec, axis=1, keepdims=True)
    vec_unit = vec / radii
    ang = np.arccos((look_at * vec_unit).sum(axis=-1, keepdims=True))
    ang_valid = ang < ang_tol
    print(f"Fraction of images looking at the center: {ang_valid.sum()/num_images:.2f}.")

    radius_mean = radii.mean()
    radii_valid = np.isclose(radius_mean, radii, rtol=radii_tol)
    print(f"Fraction of images positioned around the center: {radii_valid.sum()/num_images:.2f}.")

    valid = ang_valid * radii_valid
    print(f"Valid fraction of concentric images: {valid.sum()/num_images:.2f}.")

    return valid.sum() / num_images > pose_tol


def export_to_json(trans, scale, scene_path, file_name, split_dict=None, do_split=False):
    out = {
        "trans": trans,
        "scale": scale,
    }

    if do_split:
        if split_dict is None:
            image_names = os.listdir(os.path.join(scene_path, "images"))
            image_names = ['{:06}'.format(int(i.split(".")[0])) for i in image_names]
            split_dict = split_data(image_names, split=10)
        
        out.update(split_dict)

    with open(os.path.join(scene_path, file_name), "w") as outputfile:
        json.dump(out, outputfile, indent=4)

    return


def data_to_json(args):
    cameras, images, points3D = read_model(os.path.join(args.data_dir, "sparse"), ext=".bin")

    # define bounding regions based on scene type
    if args.scene_type == "outdoor":
        if check_concentric(images):
            center, scale, bounding_box = bound_by_pose(images)
        else:
            center, scale, bounding_box = bound_by_points(points3D)
    elif args.scene_type == "indoor":
        # use sfm points as a proxy to define bounding regions
        center, scale, bounding_box = bound_by_points(points3D)
    elif args.scene_type == "object":
        # use poses as a proxy to define bounding regions
        center, scale, bounding_box = bound_by_pose(images)
    else:
        raise TypeError("Unknown scene type")

    # export json file
    export_to_json(list(center), scale, args.data_dir, "meta.json")
    print("Writing data to json file: ", os.path.join(args.data_dir, "meta.json"))
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data")
    parser.add_argument(
        "--scene_type",
        type=str,
        default="outdoor",
        choices=["outdoor", "indoor", "object"],
        help="Select scene type. Outdoor for building-scale reconstruction; "
        "indoor for room-scale reconstruction; object for object-centric scene reconstruction.",
    )
    args = parser.parse_args()
    data_to_json(args)
