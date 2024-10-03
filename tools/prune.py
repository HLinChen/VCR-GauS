import torch

from gaussian_renderer import count_render, visi_acc_render


def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list


def prune_list(gaussians, viewpoint_stack, pipe, background):
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

        
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        
    return gaussian_list, imp_list


v_render = visi_acc_render
def get_visi_list(gaussians, viewpoint_stack, pipe, background):
    out = {}
    gaussian_list = None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = v_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list = render_pkg["countlist"]
    
    for i in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = v_render(viewpoint_cam, gaussians, pipe, background)
        gaussians_count = render_pkg["countlist"].detach()
        gaussian_list += gaussians_count
        
    visi = gaussian_list > 0
        
    out["visi"] = visi
    return out

