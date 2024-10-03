from pytorch3d.ops import ball_query, knn_points


def remove_radius_outlier(xyz, nb_points=5, radius=0.1):
    if xyz.dim() == 2: xyz = xyz[None]
    nn_dists, nn_idx, nn = ball_query(xyz, xyz, K=nb_points+1, radius=radius)
    valid = ~(nn_idx[0]==-1).any(-1)
    
    return valid


def remove_statistical_outlier(xyz, nb_points=20, std_ratio=20.):
    if xyz.dim() == 2: xyz = xyz[None]
    nn_dists, nn_idx, nn = knn_points(xyz, xyz, K=nb_points, return_sorted=False)
    
    # Compute distances to neighbors
    distances = nn_dists.squeeze(0)  # Shape: (N, nb_neighbors)

    # Compute mean and standard deviation of distances
    mean_distances = distances.mean(dim=-1)
    std_distances = distances.std(dim=-1)

    # Identify points that are not outliers
    threshold = mean_distances + std_ratio * std_distances
    valid = (distances <= threshold.unsqueeze(1)).any(dim=1)
    
    return valid


if __name__ == '__main__':
    import torch
    import time
    
    gpu = 0
    device = torch.device('cuda:{:d}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    t1 = time.time()
    xyz = torch.rand(int(1e7), 3).to(device)
    remove_statistical_outlier(xyz)
    print('time:', time.time()-t1, 's')

