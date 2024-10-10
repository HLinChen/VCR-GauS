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

import math
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

try:
    from scene.cameras import Camera
except ImportError:
    pass
from tools.general_utils import PILtoTorch, NumpytoTorch
from tools.graphics_utils import fov2focal
from tools.math_utils import inv_normalize_pts
from scene.cameras import SampleCam

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution) / 255.

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    depth = None
    if cam_info.depth is not None:
        size = list(resolution)[::-1]
        depth = NumpytoTorch(cam_info.depth, size)
    normal = None
    if cam_info.normal is not None:
        size = list(resolution)[::-1]
        normal = NumpytoTorch(cam_info.normal, size).permute(1, 2, 0)   # H, W, 3
    mask = None
    if cam_info.mask is not None:
        mask = PILtoTorch(cam_info.mask, resolution).squeeze(0)
        if mask.dim() == 3: mask = mask[0]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth=depth, normal=normal, mask=mask)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos), desc="Processing data", leave=False):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def find_up_axis(R):
    '''
    R: world to bounding box coordinate system
    '''
    
    up_vector = torch.tensor([0, -1, 0], dtype=torch.float32, device=R.device) # world colmap
    up_vector = R @ up_vector                                      # bounding box coordinate system
    up_axis = torch.argmax(torch.abs(up_vector))
    up_sign = torch.sign(up_vector[up_axis])
    
    return up_axis, up_sign


def find_axis(R, axis_name='up'):
    '''
    colmap coordinate system
    R: world to bounding box coordinate system
    '''
    if axis_name == 'up':
        axis_w=[0, -1, 0]
    elif axis_name == 'front':
        axis_w=[0, 0, 1]
    elif axis_name == 'right':
        axis_w=[1, 0, 0]
    else:
        raise ValueError(f'axis_name: "{axis_name}" should be one of [up, front, right]')
    axis_w = torch.tensor(axis_w, dtype=torch.float32, device=R.device) # world colmap
    axis_c = R @ axis_w                                      # bounding box coordinate system
    axis = torch.argmax(torch.abs(axis_c))
    sign = torch.sign(axis_c[axis])
    
    return axis, sign

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at_np(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z colmap
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector)) # z x up
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector)) # up x z
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1) # axis=1 !!!!! 把行拼起来了 w2c
    return R


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=campos.device)
    if campos.dim() == 2: up_vector = up_vector[None, :]
    if not opengl:
        # camera forward aligns with -z colmap
        forward_vector = safe_normalize(target - campos)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector)) # z x up
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        right_vector = safe_normalize(torch.cross(up_vector, forward_vector)) # up x z
        up_vector = safe_normalize(torch.cross(forward_vector, right_vector))
    R = torch.stack([right_vector, up_vector, forward_vector], dim=1) # axis=1 !!!!! 把行拼起来了 w2c
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at_np(campos, target, opengl)                             # ??? should be look_at(campos, target, opengl).transpose(0, 2, 1)
    T[:3, 3] = campos
    return T


def cubic_camera(n, trans, scale, target=None, opengl=False):
    xyz = np.random.rand(n, 3) * 2 - 1
    for i in range(3): xyz[i::3, i] = xyz[i::3, i] / np.abs(xyz[i::3, i]) # Unit cube
    
    if target is None: target = np.zeros([1, 3], dtype=np.float32)
    
    xyz = inv_normalize_pts(xyz, trans, scale)
    target = inv_normalize_pts(target, trans, scale)
    
    T = np.zeros((n, 4, 4))
    up_vector = [1, 0, 0]
    T[:, :3, :3] = look_at(xyz, target, opengl, up_vector) # c2w
    T[:, :3, 3] = xyz
    T[:, 3, 3] = 1
    
    T = np.linalg.inv(T) # w2c
    
    return T


def check_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.float32)
    else: return x


def up_camera(n, trans, scale, target=None, opengl=False): # colmap
    trans = check_tensor(trans)
    scale = check_tensor(scale)
    device = trans.device
    
    up_axis, up_sign = find_up_axis(trans[:3, :3])
    v_axis = [i for i in [0, 1, 2] if i != up_axis]
    
    xyz = torch.rand(n, 3).to(device) * 2 - 1
    xyz[:, up_axis] = up_sign # up
    
    if target is None:
        target = check_tensor(target)
        target = torch.zeros([1, 3], dtype=torch.float32, device=device)
    
    target[:, up_axis] = 1 * -up_sign # 5
    
    xyz = inv_normalize_pts(xyz, trans, scale)
    target = inv_normalize_pts(target, trans, scale)
    
    T = torch.zeros((xyz.shape[0], 4, 4), device=device)      # w2c
    R = look_at(xyz, target, opengl) # w2c
    T[:, :3, :3] = R
    T[:, :3, 3] = - (R @ xyz[..., None]).squeeze(-1) # w2c
    T[:, 3, 3] = 1
    
    return T


def around_camera(n, trans, scale, height=None, target=None, opengl=False):
    trans = check_tensor(trans)
    scale = check_tensor(scale)
    
    device = trans.device
    grid_points = torch.Tensor([
        [-1, -1, -1],
        [1, 1, 1],
    ]).to(device)
    
    up_axis, up_sign = find_up_axis(trans[:3, :3])
    v_axis = [i for i in [0, 1, 2] if i != up_axis]
    
    xyz = torch.rand(n, 3).to(device) * 2 - 1
    for i in v_axis: xyz[i-1::2, i] = xyz[i-1::2, i] / torch.abs(xyz[i-1::2, i])
    
    if target is None:
        target = check_tensor(target)
        target = torch.zeros([1, 3], dtype=torch.float32, device=device)
    
    xyz = inv_normalize_pts(xyz, trans, scale)
    target = inv_normalize_pts(target, trans, scale)
    grid_points = inv_normalize_pts(grid_points, trans, scale)
    
    if height is None: height = target[0, 1]
    
    xyz[:, 1] = height
    
    T = torch.zeros((xyz.shape[0], 4, 4), device=device)      # w2c
    R = look_at(xyz, target, opengl) # w2c
    T[:, :3, :3] = R
    T[:, :3, 3] = - (R @ xyz[..., None]).squeeze(-1) # w2c
    T[:, 3, 3] = 1
    
    return T


def bb_camera(n, trans, scale, height=None, target=None, opengl=False, up=True, around=True, look_mode='target', sample_mode='grid', boundary=0.9, bidirect=False): # colmap 0.8
    trans = check_tensor(trans)
    scale = check_tensor(scale)
    device = trans.device
    
    rot = trans[:3, :3] if trans.ndim == 2 else torch.eye(3, device=device)
    up_axis, up_sign = find_axis(rot, axis_name='up')
    if sample_mode == 'grid' or (up and around):
        right_axis, right_sign = find_axis(rot, axis_name='right')
        front_axis, front_sign = find_axis(rot, axis_name='front')
    v_axis = [i for i in [0, 1, 2] if i != up_axis]
    
    up_n = around_n = n
    if up and around:
        h = scale[up_axis]
        l = scale[right_axis]
        w = scale[front_axis]
        
        around_area = 2 * (l * h + h * w)
        up_area = l * w
        total_area = around_area + up_area
        
        up_n = int((n * up_area / total_area) * 1)
    
    xyz = []
    if target is None:
        if look_mode == 'target':
            target = torch.zeros([1, 3], dtype=torch.float32, device=device)
            target[:, up_axis] = 1 * -up_sign # 5
        else:
            target = []
    else:
        target = check_tensor(target)
    
    if up:
        if sample_mode == 'random':
            xyz_up = torch.rand(up_n, 3).to(device) * 2 - 1
        elif sample_mode == 'grid':
            xyz_up = up_grid_posi(up_n, scale, right_axis, up_axis, front_axis).to(device)
            around_n = n - up_n
        xyz_up[:, up_axis] = up_sign
        xyz.append(xyz_up)
        
        if look_mode == 'direction':
            tgt_up = xyz_up.clone()
            tgt_up[:, up_axis] *= -1
            target.append(tgt_up)
    
    if around:
        if sample_mode == 'random':
            xyz_around = torch.rand(around_n, 3).to(device) * 2 - 1
        elif sample_mode == 'grid':
            if not bidirect:
                xyz_around = around_grid_posi(around_n, scale, right_axis, up_axis, front_axis, up_sign=up_sign).to(device)
            else:
                n1 = around_n // 2
                xyz1 = around_grid_posi(n1, scale, right_axis, up_axis, front_axis, sign=1, up_sign=up_sign).to(device)
                n2 = around_n - xyz1.shape[0]
                xyz2 = around_grid_posi(n2, scale, right_axis, up_axis, front_axis, sign=-1, up_sign=up_sign).to(device)
                xyz_around = torch.cat([xyz1, xyz2], 0)
                n_trg = xyz_up.shape[0] + xyz_around.shape[0] if up else xyz_around.shape[0]
                target = target.repeat(n_trg, 1)
                target[-xyz2.shape[0]:, up_axis] *= -1
                
        xyz_around[:, up_axis] = xyz_around[:, up_axis] * boundary + (1 - boundary) * up_sign
        xyz.append(xyz_around)
        
        if look_mode == 'direction':
            trg_around = xyz_around.clone()
            for i in v_axis: trg_around[i-1::2, i] *= -1
            target.append(trg_around)
    
    xyz = torch.cat(xyz, 0)
    if look_mode == 'direction':
        target = torch.cat(target, 0)
    
    xyz = inv_normalize_pts(xyz, trans, scale)
    target = inv_normalize_pts(target, trans, scale)
    
    T = torch.zeros((xyz.shape[0], 4, 4), device=device)      # w2c
    R = look_at(xyz, target, opengl) # w2c
    T[:, :3, :3] = R
    T[:, :3, 3] = - (R @ xyz[..., None]).squeeze(-1) # w2c
    T[:, 3, 3] = 1
    
    return T


def around_grid_posi(num_points, scale, right_axis, up_axis, front_axis, sign=1, up_sign=1):
    device = scale.device
    indexing = 'xy'
    h = scale[up_axis]
    l = scale[right_axis]
    w = scale[front_axis]
    
    total_area = 2 * (l * h + h * w)
    ratio = (num_points / total_area).sqrt()
    h_points = torch.round(h * ratio).int()
    l_points = torch.round(l * ratio).int()
    w_points = torch.round(w * ratio).int()
    
    total_points = []
    h_coord = torch.arange(start=-1, end=1, step=2 / h_points, device=device) * up_sign
    
    step = 2 / l_points
    st = -1 if sign == 1 else -1 + step
    l_coord = torch.arange(start=st, end=1, step=step, device=device) # * sign
    grid_l, grid_h = torch.meshgrid([l_coord, h_coord], indexing=indexing)
    lh = torch.stack([grid_l.flatten(), grid_h.flatten()], dim=1)
    points = torch.ones([lh.shape[0], 3], dtype=torch.float32, device=device) * 1
    points[:, [right_axis, up_axis]] = lh
    total_points.append(points)
    
    # back
    step = - 2 / l_points
    st = 1 if sign == 1 else 1 + step
    l_coord = torch.arange(start=st, end=-1, step=step, device=device) # * sign
    grid_l, grid_h = torch.meshgrid([l_coord, h_coord], indexing=indexing)
    lh = torch.stack([grid_l.flatten(), grid_h.flatten()], dim=1)
    points = torch.ones([lh.shape[0], 3], dtype=torch.float32, device=device) * -1
    points[:, [right_axis, up_axis]] = lh
    total_points.append(points)
    
    # right
    step = - 2 / w_points
    st = 1 if sign == 1 else 1 + step
    w_coord = torch.arange(start=st, end=-1, step=step, device=device)
    grid_h, grid_w = torch.meshgrid([h_coord, w_coord], indexing=indexing)
    hw = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
    points = torch.ones([hw.shape[0], 3], dtype=torch.float32, device=device) * 1
    points[:, [up_axis, front_axis]] = hw
    total_points.append(points)
    
    # left
    step = 2 / w_points
    st = -1 if sign == 1 else -1 + step
    w_coord = torch.arange(start=st, end=1, step = step, device=device)
    grid_h, grid_w = torch.meshgrid([h_coord, w_coord], indexing=indexing)
    hw = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
    points = torch.ones([hw.shape[0], 3], dtype=torch.float32, device=device) * -1
    points[:, [up_axis, front_axis]] = hw
    total_points.append(points)
    
    points = torch.cat(total_points, 0)
    return points


def up_grid_posi(num_points, scale, right_axis, up_axis, front_axis):
    h = scale[up_axis]
    l = scale[right_axis]
    w = scale[front_axis]
    
    total_area = l * w
    ratio = math.sqrt(num_points / total_area)
    l_points = torch.round(l * ratio).int()
    w_points = torch.round(w * ratio).int()
    
    # up
    l_coord = torch.linspace(start=-1, end=1, steps=l_points) # * 0.9
    w_coord = torch.linspace(start=-1, end=1, steps=w_points) # * 0.9
    grid_l, grid_w = torch.meshgrid([l_coord, w_coord], indexing='xy')
    lw = torch.stack([grid_l.flatten(), grid_w.flatten()], dim=1)
    points = torch.ones([lw.shape[0], 3], dtype=torch.float32) * 1
    points[:, [right_axis, front_axis]] = lw
    
    return points


def grid_camera(trans, scale, opengl=False):
    trans = check_tensor(trans)
    scale = check_tensor(scale)
    device = trans.device
    
    xyz = torch.tensor(
        [
            [-1, -1, -1],
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
        ], dtype=torch.float32, device=device
    )
    
    
    if target is None:
        target = check_tensor(target)
        target = torch.zeros([1, 3], dtype=torch.float32, device=device)
    
    xyz = inv_normalize_pts(xyz, trans, scale)
    target = inv_normalize_pts(target, trans, scale)
    
    T = torch.zeros((xyz.shape[0], 4, 4), device=device)      # w2c
    R = look_at(xyz, target, opengl) # w2c
    T[:, :3, :3] = R
    T[:, :3, 3] = - (R @ xyz[..., None]).squeeze(-1) # w2c
    T[:, 3, 3] = 1
    
    return T


def sample_cameras(model, n, up=False, around=True, look_mode='target', sample_mode='grid', bidirect=True):
    cam_height = None
    w2cs = bb_camera(n, model.trans, model.scale, cam_height, up=up, around=around, \
        look_mode=look_mode, sample_mode=sample_mode, bidirect=bidirect)
    # traincam = self.scene.getTrainCameras()[0]
    # FoVx = traincam.FoVx        # 1.3990553440909452
    # FoVy = traincam.FoVy        # 0.8764846384037163
    # width = traincam.image_width    # 1500
    # height = traincam.image_height  # 835
    FoVx = FoVy = 2.5 # 3.14 / 2
    width = height = 1500
    cams = []
    
    for i in range(w2cs.shape[0]):
        w2c = w2cs[i]
        cam = SampleCam(w2c, width, height, FoVx, FoVy)
        cams.append(cam)
    
    return cams


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])


