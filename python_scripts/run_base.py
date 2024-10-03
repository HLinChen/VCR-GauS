import os
import time
import GPUtil


def worker(gpu, scene, factor, fn):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    fn(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor, excluded_gpus, fn):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job, fn)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


def check_finish(scene, path, type='mesh'):
    if not os.path.exists(path):
        print(f"Scene \033[1;31m{scene}\033[0m failed in \033[1;31m{type}\033[0m")
        return False
    return True


train_cmd = "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} \
        python train.py \
            --config=configs/{dataset}/{cfg}.yaml \
            --logdir={log_dir} \
            --model.source_path={data_dir}/{scene}/ \
            --train.debug_from={debug_from} \
            --model.data_device={data_device} \
            --model.resolution={resolution} \
            --wandb \
            --wandb_name {project}"


train_cmd_new = "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} \
        python train.py \
            --config={cfg} \
            --logdir={log_dir} \
            --model.source_path={data_dir}/{scene}/ \
            --train.debug_from={debug_from} \
            --model.data_device={data_device} \
            --model.resolution={resolution} \
            --wandb \
            --wandb_name {project}"


extract_mesh_cmd = "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} \
        python tools/depth2mesh.py \
                --mesh_name {ply} \
                --split {step} \
                --method {fuse_method} \
                --voxel_size {voxel_size} \
                --num_cluster {num_cluster} \
                --max_depth {max_depth} \
                --clean \
                --prob_thres {prob_thr} \
                --cfg_path {log_dir}/config.yaml"


eval_tnt_cmd = "OMP_NUM_THREADS={num_threads} CUDA_VISIBLE_DEVICES={gpu} \
                conda run -n {eval_env} \
                python evaluation/tnt_eval/run.py \
                    --dataset-dir {data_dir}/{scene}/ \
                    --traj-path {data_dir}/{scene}/{scene}_COLMAP_SfM.log \
                    --ply-path {log_dir}/{ply} > {log_dir}/fscore.txt"


eval_cd_cmd = "OMP_NUM_THREADS={num_threads}  CUDA_VISIBLE_DEVICES={gpu} \
                python evaluation/eval_dtu/evaluate_single_scene.py \
                    --input_mesh {tri_mesh_path} \
                    --scan_id {scan_id} --output_dir {output_dir} \
                    --mask_dir {data_dir} \
                    --DTU {data_dir}"


render_cmd = "CUDA_VISIBLE_DEVICES={gpu} \
                python evaluation/render.py \
                    --cfg_path {log_dir}/config.yaml \
                    --iteration 30000 \
                    --skip_train"

eval_psnr_cmd = "CUDA_VISIBLE_DEVICES={gpu} \
                    python evaluation/metrics.py \
                    --cfg_path {log_dir}/config.yaml"

eval_replica_cmd = "OMP_NUM_THREADS={num_threads} CUDA_VISIBLE_DEVICES={gpu} \
                    python evaluation/replica_eval/evaluate_single_scene.py \
                        --input_mesh {tri_mesh_path} \
                        --scene {scene} \
                        --output_dir {output_dir} \
                        --data_dir {data_dir}"