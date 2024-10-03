import os
import sys
import glob
import math
import struct
import argparse
import numpy as np
import collections

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.getcwd())
from tools.general_utils import set_random_seed


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])



def get_args(test=False):
    parser = get_default_parser()

    #↓↓↓↓
    #NOTE: project-specific args
    parser.add_argument('--NNET_architecture', type=str, default='v02')
    parser.add_argument('--NNET_output_dim', type=int, default=3, help='{3, 4}')
    parser.add_argument('--NNET_output_type', type=str, default='R', help='{R, G}')
    parser.add_argument('--NNET_feature_dim', type=int, default=64)
    parser.add_argument('--NNET_hidden_dim', type=int, default=64)

    parser.add_argument('--NNET_encoder_B', type=int, default=5)

    parser.add_argument('--NNET_decoder_NF', type=int, default=2048)
    parser.add_argument('--NNET_decoder_BN', default=False, action="store_true")
    parser.add_argument('--NNET_decoder_down', type=int, default=8)
    parser.add_argument('--NNET_learned_upsampling', default=False, action="store_true")

    parser.add_argument('--NRN_prop_ps', type=int, default=5)
    parser.add_argument('--NRN_num_iter_train', type=int, default=5)
    parser.add_argument('--NRN_num_iter_test', type=int, default=5)
    parser.add_argument('--NRN_ray_relu', default=False, action="store_true")

    parser.add_argument('--loss_fn', type=str, default='AL')
    parser.add_argument('--loss_gamma', type=float, default=0.8)
    parser.add_argument('--outdir', type=str, default='/your/log/path/')
    #↑↑↑↑

    # read arguments from txt file
    assert '.txt' in sys.argv[1]
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])

    #↓↓↓↓
    #NOTE: update args
    args.exp_root = os.path.join(args.outdir, 'dsine')
    args.load_normal = True
    args.load_intrins = True
    #↑↑↑↑

    # set working dir
    exp_dir = os.path.join(args.exp_root, args.exp_name)

    args.output_dir = os.path.join(exp_dir, args.exp_id)
    return args


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def load_intrinsic_colmap(path):
    intr_dir = os.path.join(path, "sparse", "0")
    if not os.path.exists(intr_dir):
        intr_dir = os.path.join(path, "sparse")
    # support only one camera for now
    try:
        cameras_intrinsic_file = os.path.join(intr_dir, "cameras.bin")
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_intrinsic_file = os.path.join(intr_dir, "cameras.txt")
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    intrinsics = []
    for idx, key in enumerate(cam_intrinsics):
        intrinsic = np.eye(3)
        intrinsic = torch.eye(3, dtype=torch.float32)

        intr = cam_intrinsics[key]
        height = intr.height
        width = intr.width

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    
        intrinsic[0, 0] = focal_length_x # FovX
        intrinsic[1, 1] = focal_length_y # FovY
        intrinsic[0, 2] = width / 2
        intrinsic[1, 2] = height / 2
        
        intrinsics.append(intrinsic)
    
    intrinsics = torch.stack(intrinsics, axis=0)

    return intrinsics


def test_samples(args, model, intrins=None, device='cpu'):
    img_paths = glob.glob(f'{args.img_path}/*.png') + glob.glob(f'{args.img_path}/*.jpg') + glob.glob(f'{args.img_path}/*.JPG')
    img_paths.sort()

    # normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    intrin = load_intrinsic_colmap(args.intrins_path).to(device)
    os.makedirs(args.output_path, exist_ok=True)
    
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            ext = os.path.splitext(img_path)[1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            _, _, orig_H, orig_W = img.shape

            # zero-pad the input image so that both the width and height are multiples of 32
            lrtb = utils.get_padding(orig_H, orig_W)
            img = F.pad(img, lrtb, mode="constant", value=0.0)
            img = normalize(img)
            intrins = intrin.clone()
            intrins[:, 0, 2] += lrtb[0]
            intrins[:, 1, 2] += lrtb[2]

            pred_norm = model(img, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            # save to output folder
            img_name = os.path.basename(img_path)
            # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
            # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
            pred_norm_np = pred_norm.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0) # (H, W, 3) -1, 1
            
            if args.vis:
                pred_norm_np = ((pred_norm_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
                target_path = os.path.join(args.output_path, img_name.replace(ext, '.png'))
                im = Image.fromarray(pred_norm_np)
                im.save(target_path)
            else:
                target_path = os.path.join(args.output_path, img_name.replace(ext, '.npz'))
                np.savez_compressed(target_path, pred_norm_np.astype(np.float16))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='dsine', type=str, help='path to model checkpoint')
    parser.add_argument('--mode', default='samples', type=str, help='{samples}')
    parser.add_argument("--dsine_path", dest="dsine_path", help="path to rgb image")
    parser.add_argument("--img_path", dest="img_path", help="path to rgb image")
    parser.add_argument("--intrins_path", dest="intrins_path", help="path to rgb image")
    parser.add_argument("--output_path", dest="output_path", help="path to where output image should be stored")
    parser.add_argument('--vis', action='store_true', help='visualize the output')
    args = parser.parse_args()
    
    dsine_path = args.dsine_path
    dsine_path = os.path.abspath(dsine_path)

    sys.path.append(dsine_path)

    # define model
    device = torch.device('cuda')
    set_random_seed(0)

    import utils.utils as utils
    from projects import get_default_parser
    from models.dsine.v02 import DSINE_v02 as DSINE
    
    cfg_path = f'{args.dsine_path}/projects/dsine/experiments/exp001_cvpr2024/dsine.txt'
    sys.argv = [sys.argv[0], cfg_path]
    cfg = get_args(test=True)
    
    model = DSINE(cfg).to(device)
    model.pixel_coords = model.pixel_coords.to(device)
    model = utils.load_checkpoint(args.ckpt, model)
    model.eval()
    
    # # # Load the normal predictor model from torch hub
    # model = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    
    if args.mode == 'samples':
        test_samples(args, model, intrins=None, device=device)
