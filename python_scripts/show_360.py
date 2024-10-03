import json
import numpy as np

scenes = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']

output_dirs = ["exp_360/release"]

outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

all_metrics = {"PSNR": [], "SSIM": [], "LPIPS": [], 'scene': []}
indoor_metrics = {"PSNR": [], "SSIM": [], "LPIPS": [], 'scene': []}
outdoor_metrics = {"PSNR": [], "SSIM": [], "LPIPS": [], 'scene': []}
TRIAL_NAME = 'vcr_gaus'

def show_matrix(scenes, output_dirs, TRIAL_NAME):

    for scene in scenes:
        for output in output_dirs:
            json_file = f"{output}/{scene}/{TRIAL_NAME}/results.json"
            data = json.load(open(json_file))
            data = data['ours_30000']
            
            for k in ["PSNR", "SSIM", "LPIPS"]:
                all_metrics[k].append(data[k])
                if scene in indoor_scenes:
                    indoor_metrics[k].append(data[k])
                else:
                    outdoor_metrics[k].append(data[k])
            all_metrics['scene'].append(scene)
            if scene in indoor_scenes:
                indoor_metrics['scene'].append(scene)
            else:
                outdoor_metrics['scene'].append(scene)

    latex = []
    for k in ["PSNR", "SSIM", "LPIPS"]:
        numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
        numbers = [numbers]
        if k == "PSNR":
            numbers = [f"{x:.2f}" for x in numbers]
        else:
            numbers = [f"{x:.3f}" for x in numbers]
        latex.extend([k+': ', numbers[-1]+' '])

    indoor_latex = []
    for k in ["PSNR", "SSIM", "LPIPS"]:
        numbers = np.asarray(indoor_metrics[k]).mean(axis=0).tolist()
        numbers = [numbers]
        if k == "PSNR":
            numbers = [f"{x:.2f}" for x in numbers]
        else:
            numbers = [f"{x:.3f}" for x in numbers]
        indoor_latex.extend([k+': ', numbers[-1]+' '])
        
    outdoor_latex = []
    for k in ["PSNR", "SSIM", "LPIPS"]:
        numbers = np.asarray(outdoor_metrics[k]).mean(axis=0).tolist()
        numbers = [numbers]
        if k == "PSNR":
            numbers = [f"{x:.2f}" for x in numbers]
        else:
            numbers = [f"{x:.3f}" for x in numbers]
        outdoor_latex.extend([k+': ', numbers[-1]+' '])
        
    print('Outdoor scenes')
    for i in range(len(outdoor_metrics['scene'])):
        print('PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}, scene: {}'.format(outdoor_metrics['PSNR'][i], outdoor_metrics['SSIM'][i], outdoor_metrics['LPIPS'][i], outdoor_metrics['scene'][i]))
    
    print('Indoor scenes')
    for i in range(len(indoor_metrics['scene'])):
        print('PSNR: {:.3f}, SSIM: {:.3f}, LPIPS: {:.3f}, scene: {}'.format(indoor_metrics['PSNR'][i], indoor_metrics['SSIM'][i], indoor_metrics['LPIPS'][i], indoor_metrics['scene'][i]))
        
    print('Outdoor:')
    print("".join(outdoor_latex))
    print('Indoor:')
    print("".join(indoor_latex))

if __name__ == "__main__":
    show_matrix(scenes, output_dirs, TRIAL_NAME)
