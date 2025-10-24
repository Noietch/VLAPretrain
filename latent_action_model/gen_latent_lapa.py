import sys
sys.path.append("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA")

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms as T
from torchvision.utils import save_image
import os

from latent_action_model.LAPA.laq_model import LatentActionQuantization

image_size = 256
transform = T.Compose([
    T.Resize((256, 256)),
])

# build dataset
dataset = LeRobotDataset(
    "aloha_mobile_cabinet/main",
    root="datasets/lerobot/aloha_mobile_cabinet/main",
    image_transforms=transform,
    delta_timestamps={
        "observation.images.cam_high": [-1, 0],
    }
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=32,
    shuffle=True,
)


# build model
laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).cuda()
laq.load("huggingface/latent-action-pretraining/LAPA-7B-openx/laq_openx.pt")

output_dir = "."
os.makedirs(output_dir, exist_ok=True)

for batch_idx, batch in enumerate(dataloader):
    with torch.no_grad():
        # Get images: shape [batch_size, num_frames, channels, height, width]
        images = batch['observation.images.cam_high']
        
        # Save images
        batch_size, num_frames = images.shape[0], images.shape[1]
        for i in range(batch_size):
            for j in range(num_frames):
                img = images[i, j]  # shape: [C, H, W]
                # Normalize to [0, 1] if needed (assuming images are already in [0, 1] or [0, 255])
                if img.max() > 1.0:
                    img = img / 255.0
                save_image(img, os.path.join(output_dir, f"batch{batch_idx}_sample{i}_frame{j}.png"))
        
        print(f"Saved {batch_size * num_frames} images from batch {batch_idx} to {output_dir}/")
        
        permute_image = batch['observation.images.cam_high'].permute(0, 2, 1, 3, 4).cuda()
        index_batch = laq(permute_image, return_only_codebook_ids=True)
        print(index_batch)
    break