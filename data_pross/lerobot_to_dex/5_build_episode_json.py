#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_episodes_metadata(episodes_json_path):
    """从 episodes.json 加载元数据"""
    episodes_meta = {}
    with open(episodes_json_path, 'r', encoding='utf-8') as f:
        episodes = json.load(f)

    for episode in episodes:
        episode_index = episode.get('episode_index')
        tasks = episode.get('tasks', [])
        length = episode.get('length', 0)

        # 提取第一个任务（tasks 是二维数组）
        prompt = ""
        if tasks and len(tasks) > 0 and len(tasks[0]) > 0:
            prompt = tasks[0][0]

        episodes_meta[episode_index] = {
            'prompt': prompt,
            'length': length
        }

    return episodes_meta


def detect_cameras(dataset_path):
    """检测摄像头信息"""
    cameras = []

    # 首先尝试从 videos 目录检测摄像头
    videos_dir = dataset_path / "main" / "video"
    if videos_dir.exists():
        # 检查是否有 images 子目录（类型2: images 子目录结构）
        images_dir = videos_dir / "images"
        if images_dir.exists():
            # 从 images 目录中的视频文件名提取摄像头名称
            for video_file in sorted(images_dir.glob("*.mp4")):
                filename = video_file.stem
                if "observation.images." in filename and "_episode_" in filename:
                    # 提取 camera_name
                    parts = filename.split("_episode_")
                    camera_part = parts[0].replace("observation.images.", "")
                    if camera_part not in cameras:
                        cameras.append(camera_part)
        else:
            # 检查摄像头目录（类型1: 摄像头目录结构 或 类型3: 平铺结构）
            for item in sorted(videos_dir.iterdir()):
                if item.is_dir() and item.name.startswith("observation.images."):
                    # 类型1: 摄像头目录结构
                    camera_name = item.name.replace("observation.images.", "")
                    cameras.append(camera_name)
                elif item.is_file() and item.name.endswith(".mp4"):
                    # 类型3: 平铺结构
                    filename = item.stem
                    if "observation.images." in filename and "_episode_" in filename:
                        parts = filename.split("_episode_")
                        camera_part = parts[0].replace("observation.images.", "")
                        if camera_part not in cameras:
                            cameras.append(camera_part)

    if not cameras:
        # 如果从视频目录检测失败，尝试从数据文件检测
        data_dir = dataset_path / "main" / "jsonl_data"
        data_files = sorted(data_dir.rglob("*.jsonl"))

        if data_files:
            with open(data_files[0], 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)

                    for key in data.keys():
                        if key.startswith('images_'):
                            cameras.append(key)

                    break

    return sorted(cameras)


def scan_video_files(dataset_path):
    """扫描 video 文件夹，获取所有视频文件信息

    支持三种文件结构：
    1. 摄像头目录结构: video/observation.images.{camera}/episode_{index:06d}.mp4
    2. images 子目录结构: video/images/observation.images.{camera}_episode_{index:06d}.mp4
    3. 平铺结构: video/observation.images.{camera}_episode_{index:06d}.mp4

    返回格式: {
        episode_index: {
            camera_name: video_relative_path
        }
    }
    """
    video_files = {}
    videos_dir = dataset_path / "main" / "video"

    if not videos_dir.exists():
        print(f"⚠️  警告：视频目录不存在: {videos_dir}")
        return video_files

    # 检测文件结构类型
    # 类型1: 摄像头目录结构 (observation.images.* 目录)
    camera_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith("observation.images.")]

    if camera_dirs:
        # 类型1: 摄像头目录结构
        print("  检测到摄像头目录结构")
        for camera_dir in sorted(camera_dirs):
            camera_name = camera_dir.name.replace("observation.images.", "")

            # 遍历该摄像头目录下的所有视频文件
            for video_file in sorted(camera_dir.glob("*.mp4")):
                filename = video_file.stem
                try:
                    # 文件名格式: episode_{index:06d}.mp4
                    if filename.startswith("episode_"):
                        episode_str = filename.replace("episode_", "")
                        episode_index = int(episode_str)
                    else:
                        continue

                    if episode_index not in video_files:
                        video_files[episode_index] = {}

                    rel_path = video_file.relative_to(dataset_path)
                    video_files[episode_index][camera_name] = str(rel_path)
                except (ValueError, IndexError):
                    continue
    else:
        # 类型2 或 类型3: 检查是否有 images 子目录
        images_dir = videos_dir / "images"
        if images_dir.exists():
            # 类型2: images 子目录结构
            print("  检测到 images 子目录结构")
            for video_file in sorted(images_dir.glob("*.mp4")):
                filename = video_file.stem
                try:
                    # 文件名格式: observation.images.{camera}_episode_{index:06d}.mp4
                    if "observation.images." in filename and "_episode_" in filename:
                        # 提取 camera_name 和 episode_index
                        parts = filename.split("_episode_")
                        episode_str = parts[-1]
                        episode_index = int(episode_str)

                        # 提取 camera_name
                        camera_part = parts[0].replace("observation.images.", "")
                        camera_name = camera_part

                        if episode_index not in video_files:
                            video_files[episode_index] = {}

                        rel_path = video_file.relative_to(dataset_path)
                        video_files[episode_index][camera_name] = str(rel_path)
                except (ValueError, IndexError):
                    continue
        else:
            # 类型3: 平铺结构 (视频文件直接在 video 目录下)
            print("  检测到平铺结构")
            for video_file in sorted(videos_dir.glob("*.mp4")):
                filename = video_file.stem
                try:
                    # 文件名格式: observation.images.{camera}_episode_{index:06d}.mp4
                    if "observation.images." in filename and "_episode_" in filename:
                        # 提取 camera_name 和 episode_index
                        parts = filename.split("_episode_")
                        episode_str = parts[-1]
                        episode_index = int(episode_str)

                        # 提取 camera_name
                        camera_part = parts[0].replace("observation.images.", "")
                        camera_name = camera_part

                        if episode_index not in video_files:
                            video_files[episode_index] = {}

                        rel_path = video_file.relative_to(dataset_path)
                        video_files[episode_index][camera_name] = str(rel_path)
                except (ValueError, IndexError):
                    continue

    return video_files


def build_episode_jsonl(dataset_path, episodes_json_path, output_dir=None):
    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "main" / "jsonl_data"

    if output_dir is None:
        output_dir = dataset_path / "main" / "jsonl"
    else:
        output_dir = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"构建 Episode JSONL 文件")
    print(f"{'='*80}")
    print(f"数据集路径: {dataset_path}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 episodes 元数据
    print("加载 episodes 元数据...")
    episodes_meta = load_episodes_metadata(episodes_json_path)
    print(f"✓ 加载了 {len(episodes_meta)} 个 episodes")
    print()

    # 2. 检测摄像头
    print("检测摄像头...")
    cameras = detect_cameras(dataset_path)
    print(f"✓ 检测到 {len(cameras)} 个摄像头: {cameras}")
    print()

    # 3. 扫描视频文件
    print("扫描视频文件...")
    video_files = scan_video_files(dataset_path)
    print(f"✓ 找到 {len(video_files)} 个 episode 的视频文件")
    print()

    # 4. 查找所有数据文件
    print("查找数据文件...")
    data_files = sorted(data_dir.rglob("*.jsonl"))
    print(f"✓ 找到 {len(data_files)} 个数据文件")
    print()

    if not data_files:
        print("⚠️  错误：没有找到任何数据文件")
        return 0, 0

    # 5. 按 episode_index 分组处理数据
    print("处理数据文件...")
    episode_data = defaultdict(list)
    total_records = 0

    for data_file in tqdm(data_files, desc="读取数据文件"):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                episode_index = data.get('episode_index')
                frame_index = data.get('frame_index')

                if episode_index is None:
                    continue

                meta = episodes_meta.get(episode_index, {})
                prompt = meta.get('prompt', "")

                # 构建新格式的 record
                record = {
                    'state': data.get('observation.state', [])
                }

                # 添加视频信息，格式为 images_1, images_2, ...
                # 从实际扫描到的视频文件中获取路径
                episode_videos = video_files.get(episode_index, {})
                for idx, camera_name in enumerate(cameras, start=1):
                    # 使用实际扫描到的视频路径
                    video_rel_path = episode_videos.get(camera_name)
                    if video_rel_path:
                        record[f'images_{idx}'] = {
                            'type': 'video',
                            'url': video_rel_path,
                            'frame_idx': frame_index
                        }
                    else:
                        # 如果没有找到视频文件，记录警告但继续处理
                        print(f"⚠️  警告：未找到 episode {episode_index} 的 {camera_name} 视频文件")

                # 添加 prompt 和 is_robot
                record['prompt'] = prompt
                record['is_robot'] = True

                episode_data[episode_index].append(record)
                total_records += 1


    print(f"✓ 读取了 {total_records} 条记录")
    print()

    # 6. 为每个 episode 创建一个 jsonl 文件
    print("生成 episode 文件...")
    for episode_index, records in tqdm(sorted(episode_data.items()), desc="创建文件"):
        output_file = output_dir / f"episode_{episode_index:04d}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n✓ 完成！共创建 {len(episode_data)} 个 episode 文件")
    print(f"✓ 输出目录: {output_dir}")
    print(f"{'='*80}\n")

    return len(episode_data), total_records


def main():
    base_dir = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/datasets/lerobot_spec/aloha_static_thread_velcro"
    parser = argparse.ArgumentParser(
        description="构建符合指定格式的 Episode JSONL 文件"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default=base_dir,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--episodes_json",
        type=str,
        required=False,
        default=f"{base_dir}/main/meta/episodes.json",
        help="episodes.json 文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为 dataset_path/main/jsonl）"
    )

    args = parser.parse_args()

    build_episode_jsonl(
        args.dataset_path,
        args.episodes_json,
        args.output_dir
    )


if __name__ == "__main__":
    main()
