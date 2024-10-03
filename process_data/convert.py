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
import json
import logging
from argparse import ArgumentParser
import shutil
import sys
import importlib

sys.path.append(os.getcwd())


def create_init_files(pinhole_dict_file, db_file, out_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py
    # COLMAPDatabase = getattr(importlib.import_module(f'{args.colmap_path}.scripts.python.database'), 'COLMAPDatabase')
    from submodules.colmap.scripts.python.database import COLMAPDatabase  # NOQA

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} RADIAL {width} {height} {f} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        # fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, f=fx, cx=cx, cy=cy, k1=0, k2=0)
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


def main(args):
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

        ## Feature extraction
        feat_extracton_cmd = colmap_command + " feature_extractor "\
            "--database_path " + args.source_path + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + args.camera + " \
            --SiftExtraction.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

        if args.existing_pose:
            db_file = os.path.join(args.source_path, 'distorted/database.db')
            sfm_dir = os.path.join(args.source_path, 'distorted/sparse/0')
            pinhole_dict_file = os.path.join(args.source_path, 'pinhole_dict.json')
            create_init_files(pinhole_dict_file, db_file, sfm_dir)
            
        ### Bundle adjustment
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + args.source_path + "/distorted/database.db \
            --image_path "  + args.source_path + "/input \
            --output_path "  + args.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

    if not args.skip_distorting:
        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = (colmap_command + " image_undistorter \
            --image_path " + args.source_path + "/input \
            --input_path " + args.source_path + "/distorted/sparse/0 \
            --output_path " + args.source_path + "\
            --output_type COLMAP")
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

    files = os.listdir(args.source_path + "/distorted/sparse/0")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "distorted/sparse/0", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")

        # Resize images.
        os.makedirs(args.source_path + "/images_2", exist_ok=True)
        os.makedirs(args.source_path + "/images_4", exist_ok=True)
        os.makedirs(args.source_path + "/images_8", exist_ok=True)
        # Get the list of files in the source directory
        files = os.listdir(args.source_path + "/images")
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(args.source_path, "images", file)

            destination_file = os.path.join(args.source_path, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, "images_8", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

    print("Done.")


if __name__ == '__main__':
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--skip_distorting", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    parser.add_argument("--existing_pose", action='store_true')
    parser.add_argument("--colmap_path", default="submodules.colmap", type=str)
    args = parser.parse_args()

    main(args)