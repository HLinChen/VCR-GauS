import torch
import torch.nn.functional as F

from tools.graphics_utils import depth2point_cam


def get_normal_sign(normals, begin=None, end=None, trans=None, mode='origin', vec=None):
    if mode == 'origin':
        if vec is None:
            if begin is None:
                # center
                if trans is not None:
                    begin = - trans[:3, :3].T @ trans[:3, 3] \
                        if trans.ndim != 1 else trans
                else:
                    begin = end.mean(0)
                begin[1] += 1
            vec = end - begin
        cos = (normals * vec).sum(-1, keepdim=True)
    
    return cos


def compute_gradient(img):
    dy = torch.gradient(img, dim=0)[0]
    dx = torch.gradient(img, dim=1)[0]
    return dx, dy


def compute_normals(depth_map, K):
    # Assuming depth_map is a PyTorch tensor of shape [H, W]
    # K_inv is the inverse of the intrinsic matrix
    
    _, cam_coords = depth2point_cam(depth_map[None, None], K[None])
    cam_coords = cam_coords.squeeze(0).squeeze(0).squeeze(0)        # [H, W, 3]
    
    dx, dy = compute_gradient(cam_coords)
    # Cross product of gradients gives normal
    normals = torch.cross(dx, dy, dim=-1)
    normals = F.normalize(normals, p=2, dim=-1)
    return normals
    

def compute_edge(image, k=11, thr=0.01):
    dx, dy = compute_gradient(image)
    
    edge = torch.sqrt(dx**2 + dy**2)
    edge = edge / edge.max()
    
    p = (k - 1) // 2
    edge = F.max_pool2d(edge[None], kernel_size=k, stride=1, padding=p)[0]
        
    edge[edge>thr] = 1
    return edge


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad