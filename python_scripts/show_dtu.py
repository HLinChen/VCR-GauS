import os
import json
import numpy as np

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
output_dirs = ["exp_dtu/release"]
TRIAL_NAME = 'vcr_gaus'


def show_matrix_old(scenes, output_dirs, TRIAL_NAME):
    all_metrics = {"mean_d2s": [], "mean_s2d": [], "overall": []}
    print(output_dirs)

    for scene in scenes:
        print(scene,end=" ")
        for output in output_dirs:
            json_file = f"{output}/scan{scene}/test/ours_30000/tsdf/results.json"
            data = json.load(open(json_file))
            
            for k in ["mean_d2s", "mean_s2d", "overall"]:
                all_metrics[k].append(data[k])
                print(f"{data[k]:.3f}", end=" ")
            print()

    latex = []
    for k in ["mean_d2s", "mean_s2d", "overall"]:
        numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
        
        numbers = all_metrics[k] + [numbers]
        
        numbers = [f"{x:.2f}" for x in numbers]
        if k == "overall":
            latex.extend(numbers)
        
    print(" & ".join(latex))
    

def show_matrix(scenes, output_dirs, TRIAL_NAME):
    all_metrics = {"mean_d2s": [], "mean_s2d": [], "overall": [], 'scene': []}

    for scene in scenes:
        for output in output_dirs:
            json_file = f"{output}/{scene}/{TRIAL_NAME}/results.json"
            if not os.path.exists(json_file):
                print(f"Scene \033[1;31m{scene}\033[0m was not evaluated.")
                continue
            data = json.load(open(json_file))
            
            for k in ["mean_d2s", "mean_s2d", "overall"]:
                all_metrics[k].append(data[k])
            all_metrics['scene'].append(scene)

    latex = []
    for k in ["mean_d2s", "mean_s2d", "overall"]:
        numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
        
        numbers = all_metrics[k] + [numbers]
        
        numbers = [f"{x:.2f}" for x in numbers]
        latex.extend([k+': ', numbers[-1]+' '])
        
    for i in range(len(all_metrics['scene'])):
        print('d2s: {:.3f}, s2d: {:.3f}, overall: {:.3f}, scene: {}'.format(all_metrics['mean_d2s'][i], all_metrics['mean_s2d'][i], all_metrics['overall'][i], all_metrics['scene'][i]))
    
    print("".join(latex))


if __name__ == "__main__":
    show_matrix(scenes, output_dirs, TRIAL_NAME)