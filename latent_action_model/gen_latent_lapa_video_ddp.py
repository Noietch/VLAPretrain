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
from collections import defaultdict


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
    videos = []
    for p in Path(video_dir).rglob(pattern):
        if p.is_file():
            video_path = str(p)
            if "/video/" in video_path:
                videos.append(video_path)
    return sorted(videos)


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


def _build_output_path(video_file, model_type, base_output_dir, is_temp=False, start_frame=None, rank=0):
    video_path = str(video_file)

    # Replace video directory with latent_action directory
    output_path = None
    for pattern in ["/videos/", "/video/"]:
        if pattern in video_path:
            subdir = f"temp/{model_type}" if is_temp else model_type
            output_path = video_path.replace(pattern, f"/latent_action/{subdir}/")
            output_path = os.path.splitext(output_path)[0] + ".pt"
            break

    # Fallback
    if output_path is None:
        video_path_obj = Path(video_file)
        subdir = os.path.join("temp", model_type) if is_temp else model_type
        output_dir = os.path.join(base_output_dir, subdir, video_path_obj.parent.name)
        output_path = os.path.join(output_dir, f"{video_path_obj.stem}.pt")

    # For temp paths, add slice info to filename
    if is_temp and start_frame is not None:
        temp_dir = os.path.dirname(output_path)
        video_stem = os.path.splitext(os.path.basename(output_path))[0]
        output_path = os.path.join(temp_dir, f"{video_stem}_slice_{start_frame:06d}_rank{rank}.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


class MetadataManager:
    """Manage video metadata in JSONL format."""

    def __init__(self, metadata_file):
        self.metadata_file = metadata_file
        self.cache = self._load()

    def _load(self):
        """Load metadata from jsonl file."""
        metadata_dict = {}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        metadata_dict[item['video_path']] = item
        return metadata_dict

    def save(self, metadata_item):
        """Append a single metadata item to jsonl file."""
        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(metadata_item) + '\n')
        self.cache[metadata_item['video_path']] = metadata_item

    def get(self, video_path):
        """Get metadata for a video path."""
        return self.cache.get(video_path)

    def __len__(self):
        return len(self.cache)


def find_slice_files(temp_dir, model_type):
    """Find all slice files in the temp directory by recursively searching."""
    print(f"Searching for slice files in: {temp_dir}")

    slice_files = []
    pattern_suffix = f"/latent_action/temp/{model_type}"

    for root, _, files in os.walk(temp_dir):
        if root.endswith(pattern_suffix) or pattern_suffix + "/" in root:
            slice_files.extend(
                os.path.join(root, f) for f in files
                if f.endswith('.pt') and '_slice_' in f
            )

    status = f"Found {len(slice_files)} slice files" if slice_files else \
             f"No slice files found. Looking for pattern: ...{pattern_suffix}/*.pt"
    print(status)

    return sorted(slice_files)


def group_slices_by_video(slice_files):
    """Group slice files by their source video."""
    video_slices = defaultdict(list)

    for slice_file in slice_files:
        try:
            # Load slice metadata
            data = torch.load(slice_file, map_location='cpu')
            output_path = data['output_path']

            video_slices[output_path].append({
                'file': slice_file,
                'start_frame': data['start_frame'],
                'end_frame': data['end_frame'],
                'total_frames': data['total_frames'],
                'video_path': data['video_path'],
                'model_type': data['model_type']
            })
        except Exception as e:
            print(f"Error loading slice {slice_file}: {e}")
            continue

    return video_slices


def _verify_slice_coverage(slices, expected_frames, verify):
    """Verify that slices cover all expected frames."""
    covered_frames = set()
    for slice_info in slices:
        covered_frames.update(range(slice_info['start_frame'], slice_info['end_frame']))

    if len(covered_frames) != expected_frames:
        missing_frames = set(range(expected_frames)) - covered_frames
        print(f"  ⚠ Warning: Missing frames: {sorted(list(missing_frames))[:10]}... (total: {len(missing_frames)})")
        return not verify, "Missing frames"
    return True, None


def merge_video_slices(output_path, slices, verify=True):
    """Merge all slices for a single video into the final output file."""
    slices.sort(key=lambda x: x['start_frame'])
    expected_frames = slices[0]['total_frames']

    # Verify slice coverage
    success, error = _verify_slice_coverage(slices, expected_frames, verify)
    if not success:
        return False, error

    # Load and concatenate all slices
    try:
        codebook_indices = []
        for slice_info in slices:
            data = torch.load(slice_info['file'], map_location='cpu')
            codebook_indices.append(data['codebook_idx'])
        merged_codebook = torch.cat(codebook_indices, dim=0)
    except Exception as e:
        print(f"  ✗ Error loading/concatenating slices: {e}")
        return False, str(e)

    # Verify shape
    if merged_codebook.shape[0] != expected_frames:
        print(f"  ⚠ Warning: Shape mismatch. Expected {expected_frames} frames, got {merged_codebook.shape[0]}")
        if verify:
            return False, "Shape mismatch"

    # Save merged result
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_data = {
            'codebook_idx': merged_codebook,
            'num_frames': expected_frames,
            'model_type': slices[0]['model_type']
        }
        torch.save(save_data, output_path)

        if not os.path.exists(output_path):
            return False, "File not found after save"

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        return True, f"Success ({file_size:.2f} MB)"
    except Exception as e:
        return False, str(e)


def cleanup_temp_slices(slice_files, keep_temp=False):
    """Remove temporary slice files after successful merge."""
    if keep_temp:
        print("\nKeeping temporary slice files (--keep-temp flag set)")
        return

    print("\nCleaning up temporary slice files...")
    removed_count = 0
    error_count = 0
    temp_dirs = set()

    for slice_file in slice_files:
        try:
            # Collect the directory of this slice file
            temp_dirs.add(os.path.dirname(slice_file))
            os.remove(slice_file)
            removed_count += 1
        except Exception as e:
            print(f"  Error removing {slice_file}: {e}")
            error_count += 1

    print(f"Removed {removed_count} temporary files ({error_count} errors)")

    # Remove empty temp directories
    print("\nCleaning up empty temp directories...")
    removed_dirs = 0
    for temp_dir in sorted(temp_dirs, reverse=True): 
        current_dir = temp_dir
        while current_dir and '/latent_action/temp/' in current_dir:
            if os.path.exists(current_dir) and not os.listdir(current_dir):
                os.rmdir(current_dir)
                removed_dirs += 1
                print(f"  Removed empty directory: {current_dir}")
                current_dir = os.path.dirname(current_dir)

    print(f"Removed {removed_dirs} empty directories")


def save_merge_log(output_dir, model_type, results):
    """Save merge results to a log file."""
    log_file = os.path.join(output_dir, model_type, "merge_log.json")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMerge log saved to: {log_file}")


def merge_latent_slices(temp_dir, model_type, keep_temp=False, verify=True):
    """Merge latent action slices from temp directory into final output files."""
    print("\n" + "=" * 80)
    print("MERGING LATENT ACTION SLICES")
    print("=" * 80)
    print(f"Temp directory: {temp_dir}")
    print(f"Model type: {model_type}")
    print(f"Verification: {'Disabled' if not verify else 'Enabled'}")
    print()

    # Find all slice files
    print("Scanning for slice files...")
    slice_files = find_slice_files(temp_dir, model_type)

    if len(slice_files) == 0:
        print("No slice files found. Skipping merge.")
        return

    # Group slices by video
    print("\nGrouping slices by video...")
    video_slices = group_slices_by_video(slice_files)
    print(f"Found {len(video_slices)} videos to merge")

    # Merge each video
    print("\nMerging videos...")
    results = {
        'success': [],
        'failed': []
    }

    for idx, (output_path, slices) in enumerate(video_slices.items(), 1):
        video_path = slices[0]['video_path']
        print(f"\n[{idx}/{len(video_slices)}] {video_path}")
        print(f"  Output: {output_path}")
        print(f"  Slices: {len(slices)}, Total frames: {slices[0]['total_frames']}")

        success, message = merge_video_slices(output_path, slices, verify=verify)

        if success:
            print(f"  ✓ {message}")
            results['success'].append({
                'video_path': video_path,
                'output_path': output_path,
                'num_slices': len(slices),
                'total_frames': slices[0]['total_frames']
            })
        else:
            print(f"  ✗ Failed: {message}")
            results['failed'].append({
                'video_path': video_path,
                'output_path': output_path,
                'error': message
            })

    # Print summary
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Total videos: {len(video_slices)}")
    print(f"Successfully merged: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed videos:")
        for item in results['failed']:
            print(f"  - {item['video_path']}: {item['error']}")

    # Save log
    save_merge_log(temp_dir, model_type, results)

    # Cleanup temp files if all successful
    if len(results['failed']) == 0 and not keep_temp:
        cleanup_temp_slices(slice_files, keep_temp)
    elif len(results['failed']) > 0:
        print("\n⚠ Some merges failed. Keeping all temporary files for debugging.")

    print("\n" + "=" * 80)
    print("MERGE COMPLETED!")
    print("=" * 80)


def scan_video_metadata(video_files, model_type, base_output_dir, rank=0):
    """Scan all videos to get frame counts and output paths."""
    metadata_file = os.path.join(base_output_dir, model_type, "video_metadata.jsonl")
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    # Use MetadataManager to handle metadata
    metadata_mgr = MetadataManager(metadata_file)
    if rank == 0:
        print(f"Loaded {len(metadata_mgr)} existing metadata records from {metadata_file}")

    metadata = []
    new_count = 0
    if rank == 0:
        print(f"Scanning {len(video_files)} videos to get frame counts...")

    for i, vf in enumerate(video_files):
        if not os.path.exists(vf):
            continue

        output_path = _build_output_path(vf, model_type, base_output_dir)

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        # Check if metadata already exists
        cached_meta = metadata_mgr.get(vf)
        if cached_meta:
            metadata.append(cached_meta)
            if rank == 0 and (i + 1) % 10 == 0:
                print(f"  Scanned {i + 1}/{len(video_files)} videos (using cached metadata)...")
            continue

        # Scan new video
        try:
            decoder = VideoDecoder(vf)
            metadata_item = {
                'video_path': vf,
                'num_frames': len(decoder),
                'output_path': output_path
            }
            metadata.append(metadata_item)

            if rank == 0:
                metadata_mgr.save(metadata_item)
                new_count += 1

            if rank == 0 and (i + 1) % 10 == 0:
                print(f"  Scanned {i + 1}/{len(video_files)} videos...")
        except Exception as e:
            if rank == 0:
                print(f"  Error scanning {vf}: {e}")

    if rank == 0:
        print(f"Found {len(metadata)} videos to process ({new_count} newly scanned, {len(metadata) - new_count} from cache)")
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
    # Save each slice immediately to temp directory
    slice_count = 0

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

        # Save slice immediately to temp directory
        temp_slice_path = _build_output_path(video_path, model_type, base_output_dir, is_temp=True, start_frame=start_frame, rank=rank)

        try:
            slice_data_to_save = {
                'codebook_idx': codebook_idx,
                'start_frame': start_frame,
                'end_frame': slice_data['end_frame'],
                'video_path': video_path,
                'output_path': output_path,
                'total_frames': total_frames,
                'model_type': model_type
            }
            torch.save(slice_data_to_save, temp_slice_path)
            slice_count += 1

            if slice_count % 10 == 0:
                print(f"[Rank {rank}] Saved {slice_count} slices to temp directory")

        except Exception as e:
            print(f"[Rank {rank}] Error saving slice to {temp_slice_path}: {e}")
            import traceback
            traceback.print_exc()

        # Free GPU memory after processing each slice
        torch.cuda.empty_cache()

    # Wait for all processes to finish processing
    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] Finished processing, saved {slice_count} slices to temp directory")
    else:
        print(f"[Single thread] Finished processing, saved {slice_count} slices to temp directory")

    cleanup_distributed()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate latent action representations from videos")
    parser.add_argument("--video_dir", type=str, default="extract_data/video", help="Directory containing video files")
    parser.add_argument("--base_output_dir", type=str, default="extract_data/latent_action/", help="Base output directory")
    parser.add_argument("--slice_size", type=int, default=100, help="Number of frames per slice for distributed processing")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing slices, skip processing")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging after processing")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary slice files after merging")
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

    # If merge-only mode, skip processing and only merge
    if args.merge_only:
        print("Merge-only mode: Skipping video processing")
        merge_latent_slices(
            temp_dir=args.video_dir,
            model_type=model_type,
            keep_temp=args.keep_temp,
            verify=True
        )
    else:
        # Process videos
        gen_latent_action(
            model_type=model_type,
            model_path=model_config["model_path"],
            video_path=None,
            video_dir=args.video_dir,
            video_pattern="*.mp4",
            base_output_dir=args.video_dir,
            image_size=model_config["image_size"],
            window_size=5,
            batch_size=4,
            num_workers=3,
            prefetch_factor=2,
            slice_size=args.slice_size,
        )
         # Merge slices after processing (unless --no-merge is set)
        if not args.no_merge:
            merge_latent_slices(
                temp_dir=args.video_dir,
                model_type=model_type,
                keep_temp=args.keep_temp,
                verify=True
            )
