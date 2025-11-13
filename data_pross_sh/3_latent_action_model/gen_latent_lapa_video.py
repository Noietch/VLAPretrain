import sys

sys.path.append(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA"
)

import torch
import os
from pathlib import Path
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from latent_action_model.LAPA.laq_model import LatentActionQuantization
from latent_action_model.UniVLA.genie.model import ControllableDINOLatentActionModel
from tqdm import tqdm


class VideoFramesDataset(Dataset):
    """Dataset that reads and returns preprocessed frames for each video file."""

    def __init__(self, video_files, image_size):
        self.video_files = list(video_files)
        self.image_size = image_size

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        vf = self.video_files[index]
        decoder = VideoDecoder(vf)
        frames = decoder[:]  # Shape: (T, C, H, W), values in [0, 255], uint8

        # Convert to float and normalize to [0, 1]
        frames = frames.float() / 255.0

        # Resize frames to target size
        T_frames, C, H, W = frames.shape
        if H != self.image_size or W != self.image_size:
            frames_resized = []
            for i in range(T_frames):
                frame = F.resize(
                    frames[i], [self.image_size, self.image_size], antialias=True
                )
                frames_resized.append(frame)
            frames = torch.stack(frames_resized, dim=0)
        return vf, frames


def process_and_save_video_from_frames(
    frames, output_path, model, model_type="lapa", window_size=5, batch_size=32
):
    """Process preloaded frames tensor and save latent action outputs with batch inference."""
    total_frames = frames.shape[0]

    # Prepare frame pairs for batch processing
    frame_pairs = []
    for t in range(total_frames):
        delta = min(window_size, total_frames - t - 1)
        frame_pairs.append(torch.stack([frames[t], frames[t + delta]], dim=0))
    frame_pairs = torch.stack(frame_pairs, dim=0)  # Shape: (T, 2, C, H, W)

    # Batch inference
    latent_outputs = []
    num_batches = (total_frames + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="  Processing batches", total=num_batches, leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_frames)
        batch_pairs = frame_pairs[start_idx:end_idx].cuda()  # Shape: (B, 2, C, H, W)

        with torch.no_grad():
            if model_type == "lapa":
                # LAPA expects (B, C, T, H, W)
                batch_input = batch_pairs.permute(0, 2, 1, 3, 4)
                output = model(batch_input, return_only_codebook_ids=True)
            elif model_type == "univla":
                # UniVLA expects (B, T, C, H, W)
                output = model.vq_encode(batch_pairs)["indices"]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            latent_outputs.append(output.cpu())

    # Concatenate results
    codebook_idx = torch.cat(latent_outputs, dim=0)
    torch.save(
        {
            "codebook_idx": codebook_idx,
            "num_frames": total_frames,
            "model_type": model_type,
        },
        output_path,
    )


def find_videos(video_dir, pattern="*.mp4"):
    """Find all video files in a directory matching a pattern."""
    return sorted([str(p) for p in Path(video_dir).rglob(pattern) if p.is_file()])


def load_model(model_type, model_path):
    """Load model based on type."""
    if model_type == "lapa":
        print(f"Loading LAPA model from {model_path}...")
        model = LatentActionQuantization(
            dim=1024,
            quant_dim=32,
            codebook_size=8,
            image_size=256,
            patch_size=32,
            spatial_depth=8,
            temporal_depth=8,
            dim_head=64,
            heads=16,
            code_seq_len=4,
        ).cuda()
        model.load(model_path)
        model.eval()

    elif model_type == "univla":
        print(f"Loading UniVLA model from {model_path}...")
        model = ControllableDINOLatentActionModel(
            in_dim=3,
            model_dim=768,
            latent_dim=128,
            num_latents=16,
            patch_size=14,
            enc_blocks=12,
            dec_blocks=12,
            num_heads=12,
            dropout=0.0,
        ).cuda()
        ckpt = torch.load(model_path)["state_dict"]
        model.load_state_dict(
            {k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True
        )
        model.eval()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print("Model loaded!")
    return model


def _build_output_path(video_file, model_type, base_output_dir):
    video_path_obj = Path(video_file)
    output_dir = os.path.join(base_output_dir, model_type, video_path_obj.parent.name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{video_path_obj.stem}.pt")


def gen_latent_action(
    model_type,
    model_path,
    video_path,
    video_dir,
    video_pattern,
    base_output_dir,
    image_size=256,
    window_size=5,
    batch_size=32,
    num_workers=0,
    prefetch_factor=2,
):
    """Generate latent action representations from videos."""

    # ========== Processing ==========
    if video_path is not None:
        video_files = [video_path]
        print(f"Single video mode: {video_path}")
    elif video_dir is not None:
        print(f"Searching videos in: {video_dir}\nPattern: {video_pattern}")
        video_files = find_videos(video_dir, video_pattern)
        print(f"Found {len(video_files)} video(s)")
        if len(video_files) == 0:
            print("No videos found.")
            return
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {vf}")
    else:
        print("Error: Set either video_path or video_dir")
        return

    # Load model
    model = load_model(model_type, model_path)

    # Pre-compute output paths and filter existing
    tasks = []
    for vf in video_files:
        if not os.path.exists(vf):
            print(f"Warning: Not found, skipping: {vf}")
            continue
        output_path = _build_output_path(vf, model_type, base_output_dir)
        if os.path.exists(output_path):
            print(f"Exists, skip: {output_path}")
            continue
        tasks.append((vf, output_path))

    if len(tasks) == 0:
        print("Nothing to process.")
        return

    print(
        f"\nProcessing {len(tasks)} video(s) with {model_type.upper()} (batch_size={batch_size})..."
    )
    print(
        f"Using DataLoader with num_workers={num_workers}, prefetch_factor={prefetch_factor}"
    )
    dataset = VideoFramesDataset([vf for vf, _ in tasks], image_size)

    def _collate_keep(items):
        return items  # keep as list of (vf, frames)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        collate_fn=_collate_keep,
    )
    for i, batch in enumerate(tqdm(loader, desc="Processing videos", total=len(tasks), unit="video"), 1):
        # batch is a list of length 1: [(vf, frames)]
        vf, frames = batch[0]
        output_path = _build_output_path(vf, model_type, base_output_dir)
        tqdm.write(f"\n[{i}/{len(tasks)}] {vf}")
        process_and_save_video_from_frames(
            frames, output_path, model, model_type, window_size, batch_size
        )
        tqdm.write(f"Saved to: {output_path}")

    print("\n" + "=" * 80 + "\nAll videos processed!\n" + "=" * 80)


if __name__ == "__main__":
    # ========== Configuration ==========
    # Model selection: "lapa" or "univla"
    model_type = "univla"

    # Model configurations
    MODEL_CONFIGS = {
        "lapa": {
            "model_path": "huggingface/latent-action-pretraining/LAPA-7B-openx/laq_openx.pt",
            "image_size": 256,
        },
        "univla": {
            "model_path": "huggingface/qwbu/univla-latent-action-model/main/lam-stage-2.ckpt",
            "image_size": 224,
        },
    }

    # Common parameters
    window_size = 5
    batch_size = 256
    num_workers = 1  # >0 enables async video decoding via DataLoader workers
    prefetch_factor = 2

    # Video input options
    video_path = None  # Option 1: Single video
    video_dir = "/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yiyang11/workspace/VLAPretrain/datasets/Dexmal/libero/libero"  # Option 2: Multiple videos
    video_pattern = "*.mp4"  # Pattern: "*.mp4", "*_eye.mp4", "demo_*.mp4"

    base_output_dir = "extract_data/latent_action/"

    # Get model config
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[model_type]

    gen_latent_action(
        model_type=model_type,
        model_path=model_config["model_path"],
        video_path=video_path,
        video_dir=video_dir,
        video_pattern=video_pattern,
        base_output_dir=base_output_dir,
        image_size=model_config["image_size"],
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
