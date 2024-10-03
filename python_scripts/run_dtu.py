# training scripts for the TNT datasets
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
from python_scripts.run_base import dispatch_jobs, train_cmd, extract_mesh_cmd, eval_cd_cmd, check_finish
from python_scripts.show_dtu import show_matrix

TRIAL_NAME = 'vcr_gaus'
PROJECT = 'vcr_gaus'
PROJECT_wandb = 'vcr_gaus_dtu'
DATASET = 'dtu'
base_dir = "/your/path"
output_dir = f"{base_dir}/output/{PROJECT}/{DATASET}"
data_dir = f"{base_dir}/data/DTU_mask"

do_train = False
do_extract_mesh = False
do_cd = True
dry_run = False

node = 0
max_workers = 15
be = node*max_workers
excluded_gpus = set([])

total_list = [
        'scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 
        'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122'
]
training_list = [
        'scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 
        'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122'
]

training_list = training_list[be: be + max_workers]
scenes = training_list

factors = [-1] * len(scenes)
debug_from = -1

eval_env = 'pt'
data_device = 'cuda'
voxel_size = 0.004
step = 1
PLY = f"ours.ply"
TOTAL_THREADS = 64
NUM_THREADS = TOTAL_THREADS // max_workers
prob_thr = 0.15
num_cluster = 1
max_depth = 3
fuse_method = 'tsdf_cpu'

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
                # fusion
                if do_extract_mesh:
                    if not check_finish(scene, f"{log_dir}/point_cloud", 'train'): return False
                    cmd = extract_mesh_cmd.format(gpu=gpu, ply=PLY, step=step,  fuse_method=fuse_method, voxel_size=voxel_size, num_cluster=num_cluster, max_depth=max_depth, log_dir=log_dir, prob_thr=prob_thr)
                    fail = os.system(cmd)
                    print(cmd)
        
                # evaluation
                # evaluate the mesh
                scan_id = scene[4:]
                cmd = eval_cd_cmd.format(num_threads=NUM_THREADS, gpu=gpu, tri_mesh_path=f'{log_dir}/{PLY}', scan_id=scan_id, output_dir=log_dir, data_dir=data_dir)
                if fail == 0:
                    if not dry_run:
                        if do_cd: 
                            if not check_finish(scene, f"{log_dir}/{PLY}", 'mesh'): return False
                            print(cmd)
                            fail = os.system(cmd)
                            if not check_finish(scene, f"{log_dir}/results.json", 'cd'): return False
    return fail == 0


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers) as executor:
    dispatch_jobs(jobs, executor, excluded_gpus, train_scene)

show_matrix(total_list, [output_dir], TRIAL_NAME)
print(TRIAL_NAME, " done")