import json
import os
from pathlib import Path
from tqdm import tqdm


def process_jsonl_file(jsonl_path, dataset_base_dir, latent_action_dir, model_type, output_suffix):
    """Process single JSONL file and add latent_action_* fields."""
    print(f"\nProcessing: {jsonl_path}")
    
    # Load JSONL
    with open(jsonl_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    if not lines:
        print("Empty file, skipping...")
        return
    
    # Get all images_* keys
    image_keys = sorted([k for k in lines[0].keys() if k.startswith("images_")])
    print(f"Found {len(image_keys)} image field(s): {image_keys}")
    
    # Map image keys to .pt file paths
    video_to_pt = {}
    missing = []
    
    for key in image_keys:
        if lines[0][key]["type"] == "video":
            video_url = lines[0][key]["url"]
            
            # Find corresponding .pt file
            video_path = Path(video_url)
            pt_path = Path(dataset_base_dir) / latent_action_dir / model_type / video_path.parent / f"{video_path.stem}.pt"
            
            if pt_path.exists():
                video_to_pt[key] = os.path.relpath(pt_path, dataset_base_dir)
                print(f"✓ {key} -> {video_to_pt[key]}")
            else:
                missing.append(video_url)
                print(f"⚠️  Missing .pt for: {video_url}")
    
    if missing:
        print(f"\n⚠️  {len(missing)} missing file(s). Run gen_latent_lapa_video.py first.")
        if input("Continue? (y/n): ").lower() != 'y':
            return
    
    # Add latent_action_* fields
    updated_lines = []
    for line in lines:
        updated = line.copy()
        for key in image_keys:
            if key in video_to_pt:
                latent_key = key.replace("images_", "latent_action_")
                updated[latent_key] = {
                    "type": "pt",
                    "url": video_to_pt[key],
                    "frame_idx": line[key]["frame_idx"]
                }
        updated_lines.append(updated)
    
    # Save updated JSONL
    output_path = Path(jsonl_path).parent / f"{Path(jsonl_path).stem}{output_suffix}.jsonl"
    with open(output_path, "w") as f:
        for item in updated_lines:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved: {output_path} ({len(updated_lines)} lines)\n")


def process_all_jsonl_files(jsonl_dir, dataset_base_dir, latent_action_dir, model_type, 
                           output_suffix, skip_existing):
    """Process all JSONL files in directory."""
    jsonl_files = sorted(Path(jsonl_dir).rglob("*.jsonl"))
    
    if skip_existing:
        jsonl_files = [f for f in jsonl_files if output_suffix not in f.stem]
    
    if not jsonl_files:
        print("No JSONL files found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Files to process: {len(jsonl_files)}")
    print(f"Model: {model_type} | Dataset: {dataset_base_dir}")
    print(f"{'='*80}")
    
    for jsonl_file in tqdm(jsonl_files, desc="Processing"):
        try:
            process_jsonl_file(str(jsonl_file), dataset_base_dir, latent_action_dir, 
                             model_type, output_suffix)
        except Exception as e:
            print(f"\n❌ Error: {jsonl_file}\n{e}\n")
            continue
    
    print(f"\n{'='*80}\n✓ All files processed!\n{'='*80}\n")


if __name__ == "__main__":
    # Configuration
    model_type = "univla"  # "lapa" or "univla"
    jsonl_dir = "datasets/libero_debug/jsonl"
    dataset_base_dir = "datasets/libero_debug"
    latent_action_dir = "latent_action"
    output_suffix = "_with_latent"
    skip_existing = True
    
    process_all_jsonl_files(
        jsonl_dir, 
        dataset_base_dir, 
        latent_action_dir, 
        model_type,
        output_suffix, 
        skip_existing
    )

