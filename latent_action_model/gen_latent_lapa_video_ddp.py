import sys
sys.path.append("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA")

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
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json


class VideoFrameSliceDataset(Dataset):
    """Dataset that returns frame slices from videos for distributed processing."""

    def __init__(self, video_metadata, image_size, slice_size=100):
        """
        Args:
            video_metadata: List of dicts with keys: 'video_path', 'num_frames', 'output_path'
            image_size: Target image size for resizing
            slice_size: Number of frames per slice
        """
        self.image_size = image_size
        self.slice_size = slice_size

        # Build slice index: each item is (video_idx, start_frame, end_frame)
        self.slices = []
        for video_idx, meta in enumerate(video_metadata):
            num_frames = meta['num_frames']
            for start_frame in range(0, num_frames, slice_size):
                end_frame = min(start_frame + slice_size, num_frames)
                self.slices.append({
                    'video_idx': video_idx,
                    'video_path': meta['video_path'],
                    'output_path': meta['output_path'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'total_frames': num_frames
                })

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        slice_info = self.slices[index]
        video_path = slice_info['video_path']
        start_frame = slice_info['start_frame']
        end_frame = slice_info['end_frame']

        # Load only the required frame slice
        decoder = VideoDecoder(video_path)
        frames = decoder[start_frame:end_frame].float() / 255.0  # Shape: (T, C, H, W)

        # Resize frames to target size
        T_frames, C, H, W = frames.shape
        if H != self.image_size or W != self.image_size:
            frames = torch.stack([
                F.resize(frames[i], [self.image_size, self.image_size], antialias=True)
                for i in range(T_frames)
            ])

        return {
            'frames': frames,
            'video_path': video_path,
            'output_path': slice_info['output_path'],
            'start_frame': start_frame,
            'end_frame': end_frame,
            'total_frames': slice_info['total_frames']
        }

def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not using distributed mode")
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def process_frame_slice(frames, start_frame, total_frames, model, model_type="lapa", window_size=5, batch_size=32):
    """Process a slice of frames and return latent action outputs."""
    slice_frames = frames.shape[0]

    # Prepare frame pairs for batch processing
    frame_pairs = []
    for t in range(slice_frames):
        global_t = start_frame + t
        # Calculate the future frame index
        future_offset = min(window_size, total_frames - global_t - 1)

        if t + future_offset < slice_frames:
            # Future frame is in current slice
            future_frame = frames[t + future_offset]
        else:
            # Future frame is beyond current slice, use last frame in slice
            future_frame = frames[-1]

        frame_pairs.append(torch.stack([frames[t], future_frame], dim=0))

    frame_pairs = torch.stack(frame_pairs)  # Shape: (T, 2, C, H, W)

    # Batch inference
    latent_outputs = []
    for batch_idx in range((slice_frames + batch_size - 1) // batch_size):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, slice_frames)
        batch_pairs = frame_pairs[start_idx:end_idx].cuda()

        with torch.no_grad():
            if model_type == "lapa":
                output = model(batch_pairs.permute(0, 2, 1, 3, 4), return_only_codebook_ids=True)
            else:  # univla
                output = model.vq_encode(batch_pairs)["indices"]
            latent_outputs.append(output.cpu())

    codebook_idx = torch.cat(latent_outputs, dim=0)
    return codebook_idx, start_frame


def find_videos(video_dir, pattern="*.mp4"):
    """Find all video files in a directory matching a pattern."""
    return sorted([str(p) for p in Path(video_dir).rglob(pattern) if p.is_file()])


def load_model(model_type, model_path, local_rank):
    """Load model based on type."""
    if local_rank == 0:
        print(f"Loading {model_type.upper()} model from {model_path}...")

    if model_type == "lapa":
        model = LatentActionQuantization(
            dim=1024, quant_dim=32, codebook_size=8, image_size=256, patch_size=32,
            spatial_depth=8, temporal_depth=8, dim_head=64, heads=16, code_seq_len=4
        ).cuda()
        model.load(model_path)
    else:  # univla
        model = ControllableDINOLatentActionModel(
            in_dim=3, model_dim=768, latent_dim=128, num_latents=16, patch_size=14,
            enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.0
        ).cuda()
        ckpt = torch.load(model_path, map_location=f"cuda:{local_rank}")["state_dict"]
        model.load_state_dict({k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True)

    model.eval()
    if local_rank == 0:
        print("Model loaded!")
    return model


def _build_output_path(video_file, model_type, base_output_dir):
    """Build output path for latent action file."""
    video_path = str(video_file)

    # Replace video directory with latent_action directory
    for pattern in ["/videos/", "/video/"]:
        if pattern in video_path:
            output_path = video_path.replace(pattern, f"/latent_action/{model_type}/")
            output_path = os.path.splitext(output_path)[0] + ".pt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            return output_path

    # Fallback
    video_path_obj = Path(video_file)
    output_dir = os.path.join(base_output_dir, model_type, video_path_obj.parent.name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{video_path_obj.stem}.pt")


def scan_video_metadata(video_files, model_type, base_output_dir, rank=0):
    """Scan all videos to get frame counts and output paths."""
    metadata = []
    if rank == 0:
        print(f"Scanning {len(video_files)} videos to get frame counts...")

    for i, vf in enumerate(video_files):
        if not os.path.exists(vf):
            continue

        output_path = _build_output_path(vf, model_type, base_output_dir)

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        try:
            decoder = VideoDecoder(vf)
            num_frames = len(decoder)
            metadata.append({
                'video_path': vf,
                'num_frames': num_frames,
                'output_path': output_path
            })
            if rank == 0 and (i + 1) % 10 == 0:
                print(f"  Scanned {i + 1}/{len(video_files)} videos...")
        except Exception as e:
            if rank == 0:
                print(f"  Error scanning {vf}: {e}")

    if rank == 0:
        print(f"Found {len(metadata)} videos to process")
        total_frames = sum(m['num_frames'] for m in metadata)
        print(f"Total frames: {total_frames}")

    return metadata


def gen_latent_action(
    model_type, model_path, video_path, video_dir, video_pattern,
    base_output_dir, image_size=256, window_size=5, batch_size=32,
    num_workers=0, prefetch_factor=2, slice_size=100
):
    """Generate latent action representations from videos using distributed processing with frame slicing."""

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Get video files
    if video_path:
        video_files = [video_path]
        if rank == 0:
            print(f"Single video mode: {video_path}")
    elif video_dir:
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

    # Scan video metadata (only rank 0 does this)
    if rank == 0:
        video_metadata = scan_video_metadata(video_files, model_type, base_output_dir, rank)
        if len(video_metadata) == 0:
            print("Nothing to process.")
            cleanup_distributed()
            return
    else:
        video_metadata = None

    # Broadcast metadata to all ranks
    if dist.is_initialized():
        if rank == 0:
            metadata_str = json.dumps(video_metadata)
        else:
            metadata_str = None

        # Broadcast the metadata
        metadata_list = [metadata_str]
        dist.broadcast_object_list(metadata_list, src=0)

        if rank != 0:
            video_metadata = json.loads(metadata_list[0])

    if len(video_metadata) == 0:
        if rank == 0:
            print("Nothing to process.")
        cleanup_distributed()
        return

    if rank == 0:
        total_slices = sum((m['num_frames'] + slice_size - 1) // slice_size for m in video_metadata)
        print(f"\nProcessing {len(video_metadata)} video(s) with {model_type.upper()}")
        print(f"Slice size: {slice_size} frames, Total slices: {total_slices}")
        print(f"Batch size: {batch_size}, World size: {world_size}")
        print(f"Using DataLoader with num_workers={num_workers}, prefetch_factor={prefetch_factor}")

    # Create dataset and dataloader with frame slicing
    dataset = VideoFrameSliceDataset(video_metadata, image_size, slice_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0), pin_memory=True,
        collate_fn=lambda items: items[0], sampler=sampler
    )

    # Process frame slices assigned to this rank
    # Group results by video
    video_results = {}

    for slice_data in loader:
        frames = slice_data['frames']
        video_path = slice_data['video_path']
        output_path = slice_data['output_path']
        start_frame = slice_data['start_frame']
        total_frames = slice_data['total_frames']

        print(f"[Rank {rank}/{world_size}] Processing: {video_path} "
              f"[frames {start_frame}-{slice_data['end_frame']}/{total_frames}]")

        # Process this slice
        codebook_idx, start_frame = process_frame_slice(
            frames, start_frame, total_frames, model, model_type, window_size, batch_size
        )

        # Store results grouped by video
        if output_path not in video_results:
            video_results[output_path] = {
                'slices': [],
                'total_frames': total_frames,
                'video_path': video_path
            }

        video_results[output_path]['slices'].append({
            'codebook_idx': codebook_idx,
            'start_frame': start_frame
        })

    # Wait for all processes to finish processing
    if dist.is_initialized():
        dist.barrier()

    # Gather results from all ranks
    if dist.is_initialized():
        all_results = [None] * world_size
        dist.all_gather_object(all_results, video_results)

        # Rank 0 merges and saves results
        if rank == 0:
            print("\nMerging results from all ranks...")
            merged_results = {}
            for rank_results in all_results:
                for output_path, data in rank_results.items():
                    if output_path not in merged_results:
                        merged_results[output_path] = {
                            'slices': [],
                            'total_frames': data['total_frames'],
                            'video_path': data['video_path']
                        }
                    merged_results[output_path]['slices'].extend(data['slices'])

            # Save merged results for each video
            for output_path, data in merged_results.items():
                # Sort slices by start_frame
                data['slices'].sort(key=lambda x: x['start_frame'])

                # Concatenate all slices
                codebook_indices = torch.cat([s['codebook_idx'] for s in data['slices']], dim=0)

                # Save final result
                torch.save({
                    'codebook_idx': codebook_indices,
                    'num_frames': data['total_frames'],
                    'model_type': model_type
                }, output_path)

                print(f"Saved: {output_path} ({data['total_frames']} frames)")

            print("\n" + "=" * 80 + "\nAll videos processed!\n" + "=" * 80)
    else:
        # Non-distributed mode: save results directly
        for output_path, data in video_results.items():
            data['slices'].sort(key=lambda x: x['start_frame'])
            codebook_indices = torch.cat([s['codebook_idx'] for s in data['slices']], dim=0)

            torch.save({
                'codebook_idx': codebook_indices,
                'num_frames': data['total_frames'],
                'model_type': model_type
            }, output_path)

            print(f"Saved: {output_path} ({data['total_frames']} frames)")

        print("\n" + "=" * 80 + "\nAll videos processed!\n" + "=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate latent action representations from videos")
    parser.add_argument("--video_dir", type=str, default="extract_data/video", help="Directory containing video files")
    parser.add_argument("--base_output_dir", type=str, default="extract_data/latent_action/", help="Base output directory")
    parser.add_argument("--slice_size", type=int, default=100, help="Number of frames per slice for distributed processing")
    args = parser.parse_args()

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

    # Configuration
    model_type = "lapa"
    model_config = MODEL_CONFIGS[model_type]

    gen_latent_action(
        model_type=model_type,
        model_path=model_config["model_path"],
        video_path=None,
        video_dir=args.video_dir,
        video_pattern="*.mp4",
        base_output_dir=args.base_output_dir,
        image_size=model_config["image_size"],
        window_size=50,
        batch_size=4,
        num_workers=4,
        prefetch_factor=2,
        slice_size=args.slice_size,
    )
