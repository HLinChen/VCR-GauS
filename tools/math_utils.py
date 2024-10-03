import torch


def eps_sqrt(squared, eps=1e-17):
    """
    Prepare for the input for sqrt, make sure the input positive and
    larger than eps
    """
    return torch.clamp(squared.abs(), eps)


def ndc_to_pix(p, resolution):
    """
    Reverse of pytorch3d pix_to_ndc function
    Args:
        p (float tensor): (..., 3)
        resolution (scalar): image resolution (for now, supports only aspectratio = 1)
    Returns:
        pix (long tensor): (..., 2)
    """
    pix = resolution - ((p[..., :2] + 1.0) * resolution - 1.0) / 2
    return pix


def decompose_to_R_and_t(transform_mat, row_major=True):
    """ decompose a 4x4 transform matrix to R (3,3) and t (1,3)"""
    assert(transform_mat.shape[-2:] == (4, 4)), \
        "Expecting batches of 4x4 matrice"
    # ... 3x3
    if not row_major:
        transform_mat = transform_mat.transpose(-2, -1)

    R = transform_mat[..., :3, :3]
    t = transform_mat[..., -1, :3]

    return R, t


def to_homogen(x, dim=-1):
    """ append one to the specified dimension """
    if dim < 0:
        dim = x.ndim + dim
    shp = x.shape
    new_shp = shp[:dim] + (1, ) + shp[dim + 1:]
    x_homogen = x.new_ones(new_shp)
    x_homogen = torch.cat([x, x_homogen], dim=dim)
    return x_homogen


def normalize_pts(pts, trans, scale):
    '''
    trans: (4, 4), world to 
    '''
    if trans.ndim == 1:
        pts = (pts - trans) / scale
    else:
        pts = ((trans[:3, :3] @ pts.T + trans[:3, 3:]).T) / scale
    return pts


def inv_normalize_pts(pts, trans, scale):
    if trans.ndim == 1:
        pts = pts * scale + trans
    else:
        pts = (pts * scale[None] - trans[:3, 3:].T) @ trans[:3, :3]
    
    return pts


def get_inside_normalized(xyz, trans, scale):
    pts = normalize_pts(xyz, trans, scale)
    with torch.no_grad():
        inside = torch.all(torch.abs(pts) < 1, dim=-1)
    return inside, pts