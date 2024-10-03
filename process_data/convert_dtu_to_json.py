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
import json
from argparse import ArgumentParser
import os
import cv2
from PIL import Image, ImageFile
from glob import glob
import math
import sys
from pathlib import Path
from tqdm import tqdm
import trimesh


dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
sys.path.append(dir_path.__str__())
# from process_data.convert_data_to_json import _cv_to_gl  # noqa: E402
from process_data.convert_data_to_json import export_to_json, compute_oriented_bound  # NOQA
from submodules.colmap.scripts.python.database import COLMAPDatabase  # NOQA
from submodules.colmap.scripts.python.read_write_model import rotmat2qvec  # NOQA

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def dtu_to_json(args):
    assert args.dtu_path, "Provide path to DTU dataset"
    scene_list = os.listdir(args.dtu_path)

    test_indexes = [8, 13, 16, 21, 26, 31, 34, 56]
    for scene in tqdm(scene_list):
        scene_path = os.path.join(args.dtu_path, scene)
        if not os.path.isdir(scene_path) or 'scan' not in scene:
            continue

        # trans = [0., 0., 0.]
        # scale = 1.
        id = int(scene[4:])
        pts = trimesh.load(os.path.join(args.dtu_path, f'Points/stl/stl{id:03}_total.ply'))
        trans, scale = compute_oriented_bound(pts)
        
        out = {
            "trans": trans,
            "scale": scale,
        }

        # split_dict = None
        if args.split:
            images_names = os.listdir(os.path.join(scene_path, 'images'))
            images_names = sorted([i for i in images_names if 'png' in i])
            
            train_images = [i.split('.')[0] for i in images_names if int(i.split('.')[0]) not in test_indexes]
            test_images = [i.split('.')[0] for i in images_names if int(i.split('.')[0]) in test_indexes]
            
            train_images = sorted(train_images)
            test_images = sorted(test_images)
            
            out.update({
                    'train': train_images,
                    'test': test_images,
                    })
        
            assert len(train_images) + len(test_images) == len(images_names)
        
        file_path = os.path.join(scene_path, 'meta.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=4)
        # print('Writing data to json file: ', file_path)


def load_poses(scene_path):
    camera_param = dict(np.load(os.path.join(scene_path, 'cameras_sphere.npz')))
    images_lis = sorted(glob(os.path.join(scene_path, 'image/*.png')))
    c2ws = {}
    for idx, image in enumerate(images_lis):
        image = os.path.basename(image)

        world_mat = camera_param['world_mat_%d' % idx]
        scale_mat = camera_param['scale_mat_%d' % idx]

        # scale and decompose
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsic_param, c2w = load_K_Rt_from_P(None, P)
        c2ws[image] = c2w
    
    w, h = Image.open(os.path.join(scene_path, 'image', image)).size
    
    return c2ws, intrinsic_param, w, h
        

def convert_cam_dict_to_pinhole_dict(scene_path, pinhole_dict_file):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py
    
    c2ws, intrinsic_param, w, h = load_poses(scene_path)
    
    fx = intrinsic_param[0][0]
    fy = intrinsic_param[1][1]
    cx = intrinsic_param[0][2]
    cy = intrinsic_param[1][2]
    sk_x = intrinsic_param[0][1]
    sk_y = intrinsic_param[1][0]

    print('Writing pinhole_dict to: ', pinhole_dict_file)

    pinhole_dict = {}
    for img_name in c2ws:
        c2w = c2ws[img_name]
        W2C = np.linalg.inv(c2w)

        # params
        qvec = rotmat2qvec(W2C[:3, :3])
        tvec = W2C[:3, 3]

        params = [w, h, fx, fy, cx, cy, sk_x, sk_y,
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_name] = params
    
    with open(pinhole_dict_file, 'w') as fp:
        pinhole_dict = {k: [float(x) for x in v] for k, v in pinhole_dict.items()}
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def create_init_files(pinhole_dict_file, db_file, out_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} RADIAL {width} {height} {fx} {fy} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        cx = params[4]
        cy = params[5]
        sk_x = params[6]
        sk_y = params[7]
        qvec = params[8:12]
        tvec = params[12:15]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, k1=sk_x, k2=sk_y)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def init_colmap(args):
    assert args.dtu_path, "Provide path to DTU dataset"
    scene_list = os.listdir(args.dtu_path)
    scene_list = sorted([i for i in scene_list if 'scan' in i])

    pbar = tqdm(total=len(scene_list))
    for scene in scene_list:
        pbar.set_description(desc=f'Scene: {scene}')
        pbar.update(1)
        scene_path = os.path.join(args.dtu_path, scene)

        if not os.path.exists(f"{scene_path}/image"):
            raise Exception(f"'image` folder cannot be found in {scene_path}."
                            "Please check the expected folder structure in DATA_PREPROCESSING.md")

        # extract features
        os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --ImageReader.camera_model=RADIAL \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32 \
                --ImageReader.single_camera=true"
                  )
                # --ImageReader.camera_model=RADIAL \

        # match features
        os.system(f"colmap sequential_matcher \
                --database_path {scene_path}/database.db \
                --SiftMatching.use_gpu=true"
                  )

        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        convert_cam_dict_to_pinhole_dict(scene_path, pinhole_dict_file)

        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        # sfm_dir = os.path.join(scene_path, 'colmap')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)

        # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/image \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --clear_points 1 \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        os.system(f"colmap bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=false"
                  )
        
        # undistortion
        os.system(f"colmap image_undistorter \
            --image_path {scene_path}/image \
            --input_path {scene_path}/sparse \
            --output_path {scene_path} \
            --output_type COLMAP \
            --max_image_size 1600"
                )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dtu_path', type=str, default=None)
    parser.add_argument('--export_json', action='store_true', help='export json')
    parser.add_argument('--run_colmap', action='store_true', help='export json')
    parser.add_argument('--split', action='store_true', help='export json')

    args = parser.parse_args()

    if args.run_colmap:
        init_colmap(args)

    if args.export_json:
        dtu_to_json(args)