import wandb
import imageio
import torch
import torchvision

from matplotlib import pyplot as plt
from torchvision.transforms import functional as torchvision_F


PALETTE = [
            (0, 0, 0),
            (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
            (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
            (196, 156, 148), (23, 190, 207), (247, 182, 210), (219, 219, 141), (255, 127, 14),
            (158, 218, 229), (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163),
        ]
PALETTE = torch.tensor(PALETTE, dtype=torch.uint8)


def wandb_image(images, from_range=(0, 1)):
    images = preprocess_image(images, from_range=from_range)
    wandb_image = wandb.Image(images)
    return wandb_image


def preprocess_image(images, from_range=(0, 1), cmap="viridis"):
    min, max = from_range
    images = (images - min) / (max - min)
    images = images.detach().cpu().float().clamp_(min=0, max=1)
    if images.shape[0] == 1:
        images = get_heatmap(images, cmap=cmap)
    images = tensor2pil(images)
    return images


def wandb_sem(image, palette=PALETTE):
    image = image.detach().long().cpu()
    image = PALETTE[image].float().permute(2, 0, 1)[None]
    image = tensor2pil(image)
    wandb_image = wandb.Image(image)
    return wandb_image


def tensor2pil(images):
    image_grid = torchvision.utils.make_grid(images, nrow=1, pad_value=1)
    image_grid = torchvision_F.to_pil_image(image_grid)
    return image_grid


def get_heatmap(gray, cmap):  # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).float()  # [N,3,H,W]
    return color


def save_render(render, path):
    image = torch.clamp(render, 0.0, 1.0).detach().cpu()
    image = (image.permute(1, 2, 0).numpy() * 255).astype('uint8') # [..., ::-1]
    imageio.imsave(path, image)


