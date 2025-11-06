import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2
import io
import contextlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloder.openvla.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from dataloder.openvla.rlds.oxe.utils.droid_utils import droid_finetuning_transform
OXE_STANDARDIZATION_TRANSFORMS["droid"] = droid_finetuning_transform

DATAROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/datasets"
OXEDATA = "/mnt/hdfs/user/hadoop-llm-data/dataset/open-embodiment-X/"

def add_batch_dim(data):
    """为字典中的所有张量添加batch维度（在最前面添加维度）"""
    return tf.nest.map_structure(lambda x: x[None, ...] if isinstance(x, (tf.Tensor, np.ndarray)) else x, data)


def is_episode_processed(episode_idx, jsonl_dir):
    """检查episode是否已经被处理过"""
    episode_name = f"episode_{episode_idx:05d}"
    jsonl_name = f"{episode_name}.jsonl"
    jsonl_path = os.path.join(jsonl_dir, jsonl_name)
    return os.path.exists(jsonl_path)


def process_dataset(dataset_name, dataset_path, output_dir, fps=5, max_workers=8):
    """使用多线程并行处理数据集"""
    builder = tfds.builder_from_directory(dataset_path)
    ds = builder.as_dataset(split='train')
    
    video_dir = os.path.join(output_dir, "video")
    jsonl_dir = os.path.join(output_dir, "jsonl")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(jsonl_dir, exist_ok=True)
    
    # 尝试获取总数，如果失败就不显示
    try:
        total_episodes = builder.info.splits['train'].num_examples
        print(f"Starting processing {dataset_name} ({total_episodes} episodes) with {max_workers} workers...")
    except:
        total_episodes = None
        print(f"Starting processing {dataset_name} with {max_workers} workers...")
    
    # 使用线程池并行处理
    success_count = 0
    failed_count = 0
    skipped_count = 0
    invalid_data_count = 0
    lock = threading.Lock()
    
    def process_episode_wrapper(episode_data):
        episode, episode_idx = episode_data
        # 检查是否已经处理过
        if is_episode_processed(episode_idx, jsonl_dir):
            return "skipped", episode_idx
        
        try:
            result = extract_episode(dataset_name, episode, episode_idx, video_dir, jsonl_dir, fps)
            if result == "invalid_data":
                return "invalid_data", episode_idx
            elif result == "success":
                return "success", episode_idx
            else:
                # 如果返回了其他值或None，视为成功（兼容旧版本）
                return "success", episode_idx
        except Exception as e:
            print(f"\nError processing episode {episode_idx}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return "failed", episode_idx
    
    print(f"[DEBUG] Starting ThreadPoolExecutor with {max_workers} workers...", flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        episode_idx = 0
        
        # 使用tqdm显示进度（如果知道总数就显示，否则只显示已处理数量）
        with tqdm(total=total_episodes, desc=f"Processing {dataset_name}", unit="episode") as pbar:
            # 动态提交任务并处理结果
            for episode in ds:
                episode_idx += 1
                future = executor.submit(process_episode_wrapper, (episode, episode_idx))
                futures[future] = episode_idx
                
                # 每提交一批任务后，处理已完成的任务
                if len(futures) >= max_workers * 2:
                    done_futures = []
                    for future in list(futures.keys()):
                        if future.done():
                            done_futures.append(future)
                    
                    for future in done_futures:
                        status, ep_idx = future.result()
                        with lock:
                            if status == "success":
                                success_count += 1
                            elif status == "failed":
                                failed_count += 1
                            elif status == "skipped":
                                skipped_count += 1
                            elif status == "invalid_data":
                                invalid_data_count += 1
                        del futures[future]
                        pbar.update(1)
            
            # 处理剩余的任务
            for future in as_completed(futures.keys()):
                status, ep_idx = future.result()
                with lock:
                    if status == "success":
                        success_count += 1
                    elif status == "failed":
                        failed_count += 1
                    elif status == "skipped":
                        skipped_count += 1
                    elif status == "invalid_data":
                        invalid_data_count += 1
                pbar.update(1)
        
        print("[DEBUG] TQDM loop completed!", flush=True)
        total_episodes = episode_idx  # 更新实际处理的总数
    
    print("[DEBUG] Exited ThreadPoolExecutor context!", flush=True)
    
    # 确保tqdm完成后立即打印
    print("\n[INFO] Thread pool completed, preparing summary...", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {dataset_name} - Completed!", flush=True)
    print(f"Total episodes: {total_episodes}", flush=True)
    print(f"Newly processed: {success_count}", flush=True)
    print(f"Skipped (already exists): {skipped_count}", flush=True)
    print(f"Invalid data (no language/state/images): {invalid_data_count}", flush=True)
    print(f"Failed: {failed_count}", flush=True)
    print(f"Videos saved to: {video_dir}", flush=True)
    print(f"JSONL saved to: {jsonl_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)


def extract_episode(dataset_name, episode, episode_idx, video_dir, jsonl_dir, fps=5):
    steps = episode['steps']
    
    images_by_camera = {}
    states = []
    language = None
    
    for step in steps:
        step_with_batch = add_batch_dim(step)
        # 禁用打印输出
        with contextlib.redirect_stdout(io.StringIO()):
            processed_data = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](step_with_batch)
        
        # 提取 language instruction
        if 'language_instruction' in processed_data and language is None:
            language = processed_data['language_instruction'].numpy()[0]
        
        # 提取 proprio 作为 state (移除batch维度)
        if 'proprio' in processed_data['observation']:
            proprio = processed_data['observation']['proprio'].numpy()[0]
            states.append(proprio)
        
        # 提取所有图像数据 (判断是否为图像: shape为3或4维且dtype为uint8或float)
        for obs_key, obs_value in processed_data['observation'].items():
            obs_array = obs_value.numpy()[0]  # 移除batch维度
            # 判断是否为图像: 3维数组且最后一维是3(RGB)或1(灰度)
            if len(obs_array.shape) == 3 and obs_array.shape[-1] in [1, 3]:
                # 使用观察键名来标识相机，确保同一相机的图像始终在同一轨道
                if obs_key not in images_by_camera:
                    images_by_camera[obs_key] = []
                images_by_camera[obs_key].append(obs_array)
    
    task_name = get_task_name(language)
    if language is None or len(states) == 0 or len(images_by_camera) == 0:
        return "invalid_data"
    
    episode_name = f"episode_{episode_idx:05d}"
    jsonl_name = f"{episode_name}.jsonl"
    
    video_paths = {}
    # 为每个相机键分配一个稳定的索引，使用 image_* 格式
    for cam_idx, (cam_key, frames) in enumerate(sorted(images_by_camera.items()), 1):
        video_name = f"{episode_name}_image_{cam_idx}.mp4"
        video_path = os.path.join(video_dir, video_name)
        frame_height, frame_width = frames[0].shape[:2]
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        video_paths[f"image_{cam_idx}"] = video_name
    
    jsonl_path = os.path.join(jsonl_dir, jsonl_name)
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for frame_idx, state in enumerate(states):
            json_entry = {
                "state": state.tolist(),
                "prompt": task_name,
                "is_robot": True
            }
            
            for cam_key, video_name in video_paths.items():
                json_entry[cam_key] = {
                    "type": "video",
                    "url": video_name,
                    "frame_idx": frame_idx
                }
            
            jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
    
    return "success"


def get_task_name(language):
    if isinstance(language, bytes):
        return language.decode('utf-8').strip()
    if isinstance(language, str):
        return language.strip()
    if isinstance(language, np.ndarray) and language.size > 1:
        lang_item = language.flat[0]
        return lang_item.decode('utf-8').strip() if isinstance(lang_item, bytes) else str(lang_item).strip()
    return None


def find_dataset_path(base_dir, dataset_name):
    """
    自动查找数据集的实际路径
    会尝试常见的路径模式：
    - dataset_name/version
    - dataset_name/dataset_name/version
    """
    versions = ["1.0.0", "0.1.0"]
    
    for version in versions:
        # 尝试模式1: dataset_name/version
        path1 = os.path.join(base_dir, dataset_name, version)
        if os.path.exists(path1) and os.path.isdir(path1):
            # 检查是否包含数据集标志文件
            if os.path.exists(os.path.join(path1, "features.json")) or \
               os.path.exists(os.path.join(path1, "dataset_info.json")):
                return path1
        
        # 尝试模式2: dataset_name/dataset_name/version
        path2 = os.path.join(base_dir, dataset_name, dataset_name, version)
        if os.path.exists(path2) and os.path.isdir(path2):
            if os.path.exists(os.path.join(path2, "features.json")) or \
               os.path.exists(os.path.join(path2, "dataset_info.json")):
                return path2
    
    return None


def get_dataset_path():
    """自动发现并返回数据集路径映射"""
    # 定义需要处理的数据集名称
    oxe_dataset_names = [
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
        # "droid",
        # "fractal20220817_data",
        # "kuka",
        # "language_table",
        # "viola",
        # "berkeley_cable_routing",
        # "bc_z",
        # "cmu_stretch",
        # "fanuc_manipulation_v2",
        # "utaustin_mutex",
    ]
    
    local_dataset_names = [
        # ("bridge_orig", "bridge_dataset"),  # (显示名称, 实际目录名)
    ]
    
    paths = {}
    
    # 自动查找 OXE 数据集
    for dataset_name in oxe_dataset_names:
        dataset_path = find_dataset_path(OXEDATA, dataset_name)
        if dataset_path:
            paths[dataset_name] = dataset_path
            print(f"✓ Found {dataset_name}: {dataset_path}")
        else:
            print(f"✗ Warning: {dataset_name} not found in {OXEDATA}")
    
    # 查找本地数据集
    for display_name, dir_name in local_dataset_names:
        dataset_path = find_dataset_path(DATAROOT, dir_name)
        if dataset_path:
            paths[display_name] = dataset_path
            print(f"✓ Found {display_name}: {dataset_path}")
        else:
            print(f"✗ Warning: {display_name} not found in {DATAROOT}")
    
    return paths


if __name__ == "__main__":
    # 多线程配置
    MAX_WORKERS = 32  # 每个数据集内部并行处理episodes的线程数
    FPS = 5  # 视频帧率
    
    print(f"\n{'='*60}")
    print("Scanning and discovering dataset paths...")
    print(f"{'='*60}")
    dataset_paths = get_dataset_path()
    
    if not dataset_paths:
        print("Error: No datasets found!")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"Starting OXE Dataset Conversion (Multi-threaded)")
    print(f"Worker threads: {MAX_WORKERS}")
    print(f"Found {len(dataset_paths)} datasets to process")
    print(f"{'='*60}\n")
    
    # 串行处理数据集，每个数据集内部并行处理episodes
    for dataset_name, dataset_path in dataset_paths.items():
        output_dir = os.path.join(DATAROOT, "oxe-converted", dataset_name)
        process_dataset(dataset_name, dataset_path, output_dir, fps=FPS, max_workers=MAX_WORKERS)
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}\n")