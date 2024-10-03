import os
import numpy as np

training_list = [
    'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'
]

scenes = training_list

DATASET = 'tnt'
base_dir = "/your/log/path/"
TRIAL_NAME = 'vcr_gaus'
PROJECT = 'sq_gs'
output_dirs = [f"{base_dir}/{PROJECT}/{DATASET}"]


def show_matrix(scenes, output_dirs, TRIAL_NAME):
    all_metrics = {"precision": [], "recall": [], "f-score": [], 'scene': []}
    for scene in scenes:
        for output in output_dirs:
            # precision
            eval_file = os.path.join(output, scene, f"{TRIAL_NAME}/evaluation/evaluation.txt")
            
            if not os.path.exists(eval_file):
                print(f"Scene \033[1;31m{scene}\033[0m was not evaluated.")
                continue
            with open(eval_file, 'r') as f:
                matrix = f.readlines()
            
            precision = float(matrix[2].split(" ")[-1])
            recall = float(matrix[3].split(" ")[-1])
            f_score = float(matrix[4].split(" ")[-1])
            
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f-score"].append(f_score)
            all_metrics['scene'].append(scene)


    latex = []
    for k in ["precision","recall", "f-score"]:
        numbers = all_metrics[k]
        mean = np.mean(numbers)
        numbers = numbers + [mean]
        
        numbers = [f"{x:.3f}" for x in numbers]
        latex.extend([k+': ', numbers[-1]+' '])
        
    for i in range(len(all_metrics['scene'])):
        print('precision: {:.3f}, recall: {:.3f}, f-score: {:.3f}, scene: {}'.format(all_metrics['precision'][i], all_metrics['recall'][i], all_metrics['f-score'][i], all_metrics['scene'][i]))
    
    print("".join(latex))
    
    return


if __name__ == "__main__":
    show_matrix(scenes, output_dirs, TRIAL_NAME)
