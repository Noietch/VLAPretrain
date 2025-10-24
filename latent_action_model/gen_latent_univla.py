import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision import transforms as T

from latent_action_model.UniVLA.genie.model import ControllableDINOLatentActionModel

transform = T.Compose([
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((224, 224)),
            # T.ToTensor()
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
latent_action_model =  ControllableDINOLatentActionModel(
    in_dim=3,
    model_dim=768,
    latent_dim=128,
    num_latents=16,
    patch_size=14,
    enc_blocks=12,
    dec_blocks=12,
    num_heads=12,
    dropout=0.,
).cuda()
lam_ckpt = torch.load("huggingface/qwbu/univla-latent-action-model/main/lam-stage-2.ckpt")['state_dict']
new_ckpt = {}
for key in lam_ckpt.keys():
    new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

latent_action_model.load_state_dict(new_ckpt, strict=True)
latent_action_model = latent_action_model.cuda().eval()

for batch in dataloader:
    with torch.no_grad():
        video = batch['observation.images.cam_high'].cuda()
        latent_action_idx = latent_action_model.vq_encode(video)['indices'].squeeze()
        print(latent_action_idx.shape)
    break