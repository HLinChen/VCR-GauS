# training scripts for the TNT datasets
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
from python_scripts.run_base import dispatch_jobs, train_cmd, extract_mesh_cmd, eval_tnt_cmd, check_finish
from python_scripts.show_tnt import show_matrix


TRIAL_NAME = 'vcr_gaus'
PROJECT = 'vcr_gaus'
DATASET = 'tnt'
base_dir = "/your/path"
output_dir = f"{base_dir}/output/{PROJECT}/{DATASET}"
data_dir = f"{base_dir}/data/{DATASET}"

do_train = True
do_extract_mesh = True
do_f1 = True
dry_run = False

node = 0
max_workers = 4
be = node*max_workers
excluded_gpus = set([])

total_list = [
        'Barn', 'Caterpillar', 'Courthouse', 'Ignatius',
        'Meetingroom', 'Truck'
]
training_list = [
        'Barn', 'Caterpillar', 'Courthouse', 'Ignatius',
        'Meetingroom', 'Truck'
]

training_list = training_list[be: be + max_workers]
scenes = training_list

factors = [1] * len(scenes)
debug_from = -1 # enable wandb

eval_env = 'f1eval'
data_device = 'cpu'
step = 3
voxel_size = [0.02, 0.015, 0.01] + [x / 1000.0 for x in range(2, 10, 1)][::-1]
voxel_size = sorted(voxel_size)
PLY = f"ours.ply"
TOTAL_THREADS = 128
NUM_THREADS = TOTAL_THREADS // max_workers
prob_thr = 0.3
num_cluster = 1000
fuse_method = 'tsdf'
max_depth = 8


jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):
    time.sleep(2*gpu)
    os.system('ulimit -n 9000')
    log_dir = f"{output_dir}/{scene}/{TRIAL_NAME}"
    
    fail = 0
    
    if not dry_run:
        if do_train:
            cmd = train_cmd.format(gpu=gpu, dataset=DATASET, cfg=scene,
                                scene=scene, log_dir=log_dir, 
                                data_dir=data_dir, debug_from=debug_from, 
                                data_device=data_device, resolution=factor, project=PROJECT)
            print(cmd)
            fail = os.system(cmd)

        if fail == 0:
            if not dry_run:
                # fusion
                if do_extract_mesh:
                    if not check_finish(scene, f"{log_dir}/point_cloud", 'train'): return False
                    for vs in voxel_size:
                        cmd = extract_mesh_cmd.format(gpu=gpu, ply=PLY, step=step,  fuse_method=fuse_method, voxel_size=vs, num_cluster=num_cluster, max_depth=max_depth, log_dir=log_dir, prob_thr=prob_thr)
                        fail = os.system(cmd)
                        if fail == 0: break
                    print(cmd)
        
                # evaluation
                # You need to install open3d==0.9 for evaluation
                # evaluate the mesh
                cmd = eval_tnt_cmd.format(num_threads=NUM_THREADS, gpu=gpu, eval_env=eval_env, data_dir=data_dir, scene=scene, log_dir=log_dir, ply=PLY)
                if fail == 0:
                    if not dry_run:
                        if do_f1: 
                            if not check_finish(scene, f"{log_dir}/{PLY}", 'mesh'): return False
                            print(cmd)
                            fail = os.system(cmd)
                            if not check_finish(scene, f"{log_dir}/evaluation/evaluation.txt", 'f1'): return False
    # return True
    return fail == 0


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers) as executor:
    dispatch_jobs(jobs, executor, excluded_gpus, train_scene)

show_matrix(total_list, [output_dir], TRIAL_NAME)
print(TRIAL_NAME, " done")