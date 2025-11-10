#!/usr/bin/env python3
"""
Convert parquet files to jsonl format
将 data 文件夹下的 .parquet 文件转换为 jsonl 文件夹下的 .jsonl 文件
"""
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Recursively convert array elements
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj


def build_output_path(parquet_file, base_dir):
    parquet_path = str(parquet_file)

    if "/meta/" in parquet_path:
        output_path = parquet_path.replace("/meta/", "/jsonl_meta/")
        output_path = os.path.splitext(output_path)[0] + ".jsonl"
        return output_path
    elif "/data/" in parquet_path:
        output_path = parquet_path.replace("/data/", "/jsonl_data/")
        output_path = os.path.splitext(output_path)[0] + ".jsonl"
        return output_path
    else:
        # Fallback: create jsonl directory at the same level as data/meta
        parquet_path_obj = Path(parquet_file)
        # Find the data or meta directory in the path
        parts = parquet_path_obj.parts
        if "data" in parts:
            data_idx = parts.index("data")
            new_parts = list(parts[:data_idx]) + ["jsonl_data"] + list(parts[data_idx+1:])
            output_path = Path(*new_parts)
            output_path = output_path.with_suffix(".jsonl")
            return str(output_path)
        elif "meta" in parts:
            meta_idx = parts.index("meta")
            new_parts = list(parts[:meta_idx]) + ["jsonl_meta"] + list(parts[meta_idx+1:])
            output_path = Path(*new_parts)
            output_path = output_path.with_suffix(".jsonl")
            return str(output_path)
        else:
            # If no data/meta directory found, just change extension
            return str(parquet_path_obj.with_suffix(".jsonl"))


def find_parquet_files(base_dir, pattern="*.parquet"):
    """Find all parquet files in data and meta directories."""
    base_path = Path(base_dir)
    parquet_files = []

    # Find all parquet files under data and meta directories
    for parquet_file in base_path.rglob(pattern):
        file_str = str(parquet_file)
        # Include files that are in a 'data' or 'meta' directory
        if ("/data/" in file_str or "\\data\\" in file_str or
            "/meta/" in file_str or "\\meta\\" in file_str):
            parquet_files.append(file_str)

    return sorted(parquet_files)


def convert_parquet_to_jsonl(parquet_file, output_file, verbose=False):
    """
    Convert a single parquet file to jsonl format.
    Each row in the parquet file becomes one JSON line.
    """
    try:
        # Read parquet file
        if verbose:
            print(f"Reading: {parquet_file}")
        df = pd.read_parquet(parquet_file)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Convert each row to JSON and write to file
        with open(output_file, 'w') as f:
            for idx, row in df.iterrows():
                # Convert row to dictionary
                row_dict = row.to_dict()

                # Convert numpy types to Python native types
                row_dict = convert_numpy_types(row_dict)

                # Write as JSON line
                f.write(json.dumps(row_dict) + '\n')

        if verbose:
            print(f"Saved: {output_file} ({len(df)} rows)")

        return True, len(df)

    except Exception as e:
        print(f"Error converting {parquet_file}: {e}")
        return False, 0


def load_conversion_metadata(metadata_file):
    """Load metadata from jsonl file to track converted files."""
    metadata_dict = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    metadata_dict[item['parquet_file']] = item
    return metadata_dict


def save_conversion_metadata(metadata_file, metadata_item):
    """Append a single metadata item to jsonl file."""
    with open(metadata_file, 'a') as f:
        f.write(json.dumps(metadata_item) + '\n')


def batch_convert_parquet_to_jsonl(base_dir, force_reconvert=False):
    """
    Batch convert all parquet files in data and meta directories to jsonl format.

    Args:
        base_dir: Base directory containing parquet files
        force_reconvert: If True, reconvert even if output file exists
    """
    print(f"{'='*80}")
    print(f"Parquet to JSONL Converter")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Force reconvert: {force_reconvert}")
    print(f"{'='*80}\n")

    # Find all parquet files
    print("Searching for parquet files in data and meta directories...")
    parquet_files = find_parquet_files(base_dir)
    print(f"Found {len(parquet_files)} parquet file(s)\n")

    if len(parquet_files) == 0:
        print("No parquet files found in data or meta directories.")
        return

    # Setup metadata tracking
    metadata_file = os.path.join(base_dir, "conversion_metadata.jsonl")
    os.makedirs(os.path.dirname(metadata_file) if os.path.dirname(metadata_file) else ".", exist_ok=True)

    # Load existing metadata
    existing_metadata = load_conversion_metadata(metadata_file)
    print(f"Loaded {len(existing_metadata)} existing conversion records\n")

    # Convert files
    converted_count = 0
    skipped_count = 0
    error_count = 0
    total_rows = 0

    print(f"{'='*80}")
    print("Starting conversion...")
    print(f"{'='*80}\n")

    for i, parquet_file in enumerate(tqdm(parquet_files, desc="Converting files"), 1):
        output_file = build_output_path(parquet_file, base_dir)

        # Skip if already converted and not forcing reconversion
        if not force_reconvert and os.path.exists(output_file):
            skipped_count += 1
            print(f"[{i}/{len(parquet_files)}] Skipped (already exists): {output_file}")
            continue

        # Convert file
        print(f"[{i}/{len(parquet_files)}] Converting: {parquet_file}")
        print(f"                    → {output_file}")

        success, num_rows = convert_parquet_to_jsonl(parquet_file, output_file, verbose=False)

        if success:
            converted_count += 1
            total_rows += num_rows

            # Save metadata
            metadata_item = {
                'parquet_file': parquet_file,
                'output_file': output_file,
                'num_rows': num_rows,
                'status': 'success'
            }
            save_conversion_metadata(metadata_file, metadata_item)
            print(f"                    ✓ Success ({num_rows} rows)")
        else:
            error_count += 1
            print(f"                    ✗ Failed")

    # Print summary
    print(f"\n{'='*80}")
    print("Conversion Summary:")
    print(f"{'='*80}")
    print(f"  Total parquet files found: {len(parquet_files)}")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total rows converted: {total_rows}")
    print(f"  Metadata saved to: {metadata_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert parquet files to jsonl format"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yiyang11/workspace/VLAPretrain/datasets/lerobot_local_debug",
        help="Base directory containing parquet files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion even if output file exists"
    )

    args = parser.parse_args()

    # Run batch conversion
    batch_convert_parquet_to_jsonl(args.base_dir, force_reconvert=args.force)
