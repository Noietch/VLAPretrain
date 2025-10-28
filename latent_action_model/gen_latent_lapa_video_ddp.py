import sys

sys.path.append(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA"
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from pathlib import Path
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from latent_action_model.LAPA.laq_model import LatentActionQuantization
from latent_action_model.UniVLA.genie.model import ControllableDINOLatentActionModel
from tqdm import tqdm
import argparse


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


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


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
    for batch_idx in range(num_batches):
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


def load_model(model_type, model_path, local_rank):
    """Load model based on type and wrap with DDP."""
    if model_type == "lapa":
        if local_rank == 0:
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
        if local_rank == 0:
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
        ckpt = torch.load(model_path, map_location=f"cuda:{local_rank}")["state_dict"]
        model.load_state_dict(
            {k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True
        )
        model.eval()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if local_rank == 0:
        print("Model loaded!")

    # Note: We don't wrap with DDP since we're doing inference only
    # and distributing data across GPUs, not doing gradient updates
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
    """Generate latent action representations from videos using distributed processing."""

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # ========== Processing ==========
    if video_path is not None:
        video_files = [video_path]
        if rank == 0:
            print(f"Single video mode: {video_path}")
    elif video_dir is not None:
        if rank == 0:
            print(f"Searching videos in: {video_dir}\nPattern: {video_pattern}")
        video_files = find_videos(video_dir, video_pattern)
        if rank == 0:
            print(f"Found {len(video_files)} video(s)")
            if len(video_files) == 0:
                print("No videos found.")
                cleanup_distributed()
                return
            for i, vf in enumerate(video_files, 1):
                print(f"  {i}. {vf}")
    else:
        if rank == 0:
            print("Error: Set either video_path or video_dir")
        cleanup_distributed()
        return

    # Load model
    model = load_model(model_type, model_path, local_rank)

    # Pre-compute output paths and filter existing
    tasks = []
    for vf in video_files:
        if not os.path.exists(vf):
            if rank == 0:
                print(f"Warning: Not found, skipping: {vf}")
            continue
        output_path = _build_output_path(vf, model_type, base_output_dir)
        if os.path.exists(output_path):
            if rank == 0:
                print(f"Exists, skip: {output_path}")
            continue
        tasks.append((vf, output_path))

    if len(tasks) == 0:
        if rank == 0:
            print("Nothing to process.")
        cleanup_distributed()
        return

    if rank == 0:
        print(
            f"\nProcessing {len(tasks)} video(s) with {model_type.upper()} "
            f"(batch_size={batch_size}, world_size={world_size})..."
        )
        print(
            f"Using DataLoader with num_workers={num_workers}, prefetch_factor={prefetch_factor}"
        )

    dataset = VideoFramesDataset([vf for vf, _ in tasks], image_size)

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    def _collate_keep(items):
        return items  # keep as list of (vf, frames)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        collate_fn=_collate_keep,
        sampler=sampler,  # Use distributed sampler
    )

    # Process videos assigned to this rank
    for i, batch in enumerate(loader, 1):
        # batch is a list of length 1: [(vf, frames)]
        vf, frames = batch[0]
        output_path = _build_output_path(vf, model_type, base_output_dir)

        if rank == 0 or True:  # Print from all ranks to see progress
            print(f"[Rank {rank}/{world_size}] Processing: {vf}")

        process_and_save_video_from_frames(
            frames, output_path, model, model_type, window_size, batch_size
        )

        if rank == 0 or True:
            print(f"[Rank {rank}/{world_size}] Saved to: {output_path}")

    # Wait for all processes to finish
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 80 + "\nAll videos processed!\n" + "=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    # ========== Configuration ==========
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate latent action representations from videos")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="extract_data/video",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="extract_data/latent_action/",
        help="Base output directory for latent action files"
    )
    args = parser.parse_args()

    # Model selection: "lapa" or "univla"
    model_type = "univla"

    # Model configurations
    MODEL_CONFIGS = {
        "lapa": {
            "model_path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/huggingface/latent-action-pretraining/LAPA-7B-openx/laq_openx.pt",
            "image_size": 256,
        },
        "univla": {
            "model_path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/huggingface/qwbu/univla-latent-action-model/main/lam-stage-2.ckpt",
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
    video_dir = args.video_dir  # Option 2: Multiple videos
    video_pattern = "*.mp4"  # Pattern: "*.mp4", "*_eye.mp4", "demo_*.mp4"

    base_output_dir = args.base_output_dir

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
