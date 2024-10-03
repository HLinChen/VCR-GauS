# Training script for the Mip-NeRF 360 dataset
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
from python_scripts.run_base import dispatch_jobs, train_cmd, extract_mesh_cmd, check_finish, render_cmd, eval_psnr_cmd
from python_scripts.show_360 import show_matrix


TRIAL_NAME = 'vcr_gaus'
PROJECT = 'vcr_gaus'
PROJECT_wandb = 'vcr_gaus_360'

do_train = True
do_render = True
do_eval = True
do_extract_mesh = True
dry_run = False

node = 0
max_workers = 9
be = node*max_workers
excluded_gpus = set([])
excluded_gpus = set([2,3])

total_list = [
        "bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"
]
training_list = [
        "bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"
]

training_list = training_list[be: be + max_workers]
scenes = training_list

factors = [-1] * len(scenes)

debug_from = -1

DATASET = '360_v2'
eval_env = 'pt'
data_device = 'cpu'
step = 1
max_depth = 6.0
voxel_size = 8e-3
PLY = f"fused_mesh_split{step}.ply"
TOTAL_THREADS = 64
NUM_THREADS = TOTAL_THREADS // max_workers
prob_thr = 0.15
num_cluster = 1000
fuse_method = 'tsdf'

base_dir = "/your/path"
output_dir = f"{base_dir}/output/{PROJECT}/{DATASET}"
data_dir = f"{base_dir}/data/{DATASET}"

jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):
    time.sleep(2*gpu)
    os.system('ulimit -n 9000')
    log_dir = f"{output_dir}/{scene}/{TRIAL_NAME}"
    
    fail = 0
    
    if not dry_run:
        if do_train:
            cmd = train_cmd.format(gpu=gpu, dataset=DATASET, cfg='base',
                                scene=scene, log_dir=log_dir, 
                                data_dir=data_dir, debug_from=debug_from, 
                                data_device=data_device, resolution=factor, project=PROJECT_wandb)
            print(cmd)
            fail = os.system(cmd)
            
        if fail == 0:
            if not dry_run:
                # render
                cmd = render_cmd.format(gpu=gpu, log_dir=log_dir)
                if fail == 0:
                    if not dry_run:
                        if do_render:
                            print(cmd)
                            fail = os.system(cmd)
                            if not check_finish(scene, f"{log_dir}/test/ours_30000/renders", 'render'): return False
                
                # eval
                cmd = eval_psnr_cmd.format(gpu=gpu, log_dir=log_dir)
                if fail == 0:
                    if not dry_run:
                        if do_eval:
                            print(cmd)
                            fail = os.system(cmd)
                            if not check_finish(scene, f"{log_dir}/results.json", 'eval'): return False
                
                # fusion
                if do_extract_mesh:
                    if not check_finish(scene, f"{log_dir}/point_cloud", 'train'): return False
                    cmd = extract_mesh_cmd.format(gpu=gpu, ply=PLY, step=step,  fuse_method=fuse_method, voxel_size=voxel_size, num_cluster=num_cluster, max_depth=max_depth, log_dir=log_dir, prob_thr=prob_thr)
                    fail = os.system(cmd)
                    print(cmd)
        
    return fail == 0


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers) as executor:
    dispatch_jobs(jobs, executor, excluded_gpus, train_scene)

show_matrix(total_list, [output_dir], TRIAL_NAME)
print(TRIAL_NAME, " done")
