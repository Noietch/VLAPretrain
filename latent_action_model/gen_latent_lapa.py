import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms as T

from latent_action_model.LAPA.laq_model import LatentActionQuantization

image_size = 256
transform = T.Compose([
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((256, 256)),
            # T.ToTensor()
        ])

# build dataset
dataset = LeRobotDataset(
    "aloha_mobile_cabinet/main",
    root="datasets/lerobot/aloha_mobile_cabinet/main",
    video_backend="pyav",
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

for batch in dataloader:
    with torch.no_grad():
        permute_image = batch['observation.images.cam_high'].permute(0, 2, 1, 3, 4).cuda()
        index_batch = laq(permute_image, return_only_codebook_ids=True)
        print(index_batch)
    break