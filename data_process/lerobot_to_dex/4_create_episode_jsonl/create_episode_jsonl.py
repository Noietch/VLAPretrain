#!/usr/bin/env python3
"""
为每个episode创建一个jsonl文件，包含observation.state、tasks和视频路径信息
支持批量处理多个数据集
"""
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def find_datasets(base_dir):
    """查找所有包含jsonl_data和jsonl_meta的数据集"""
    datasets = []
    base_path = Path(base_dir)

    # 查找所有包含 jsonl_meta/episodes 的数据集
    for episodes_dir in base_path.rglob("jsonl_meta/episodes"):
        dataset_root = episodes_dir.parent.parent.parent
        data_dir = dataset_root / "main" / "jsonl_data"

        if data_dir.exists():
            datasets.append(str(dataset_root))

    return sorted(list(set(datasets)))


def load_episode_metadata_from_dir(meta_episodes_dir):
    """从指定目录加载episode元数据"""
    episodes_meta = {}

    # 查找所有jsonl文件
    jsonl_files = sorted(meta_episodes_dir.rglob("*.jsonl"))

    if not jsonl_files:
        return episodes_meta

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                meta = json.loads(line)
                episode_index = meta.get('episode_index')

                if episode_index is None:
                    continue

                # 提取tasks
                tasks = meta.get('tasks', [])

                # 提取所有摄像头的视频信息
                video_info = {}
                for key in meta.keys():
                    if key.startswith("videos/") and key.endswith("/chunk_index"):
                        camera_name = key.replace("videos/", "").replace("/chunk_index", "")

                        # 获取camera短名称（最后一部分）
                        camera_short_name = camera_name.split('.')[-1]

                        # 构建处理后的视频相对路径
                        video_rel_path = f"main/video/{camera_short_name}/observation.images.{camera_short_name}_episode_{episode_index:06d}.mp4"

                        video_info[camera_name] = {
                            'video_path': video_rel_path
                        }

                episodes_meta[episode_index] = {
                    'tasks': tasks,
                    'videos': video_info,
                    'dataset_from_index': meta.get('dataset_from_index'),
                    'dataset_to_index': meta.get('dataset_to_index'),
                    'length': meta.get('length')
                }

    return episodes_meta


def process_single_dataset(dataset_path):
    """处理单个数据集"""
    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "main" / "jsonl_data"
    meta_episodes_dir = dataset_path / "main" / "jsonl_meta" / "episodes"
    output_dir = dataset_path / "main" / "jsonl"

    print(f"\n{'='*80}")
    print(f"处理数据集: {dataset_path.name}")
    print(f"{'='*80}")
    print(f"数据目录: {data_dir}")
    print(f"元数据目录: {meta_episodes_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 1. 读取episode元数据
    print("加载episode元数据...")
    episodes_meta = load_episode_metadata_from_dir(meta_episodes_dir)

    if not episodes_meta:
        print("⚠️  错误：没有找到任何episode元数据")
        return 0, 0

    print(f"✓ 找到 {len(episodes_meta)} 个episodes")

    # 检测所有摄像头
    all_cameras = set()
    for meta in episodes_meta.values():
        all_cameras.update(meta['videos'].keys())
    print(f"✓ 检测到 {len(all_cameras)} 个摄像头: {sorted(all_cameras)}")
    print()

    # 2. 查找所有jsonl_data文件
    print("查找数据文件...")
    data_files = sorted(data_dir.rglob("*.jsonl"))
    print(f"✓ 找到 {len(data_files)} 个数据文件")
    print()

    if not data_files:
        print("⚠️  错误：没有找到任何数据文件")
        return 0, 0

    # 3. 读取所有数据文件，按episode_index分组
    print("读取观察数据...")
    episode_data = defaultdict(list)

    for data_file in data_files:
        with open(data_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                episode_index = data.get('episode_index')
                frame_index = data.get('frame_index')

                if episode_index is None:
                    continue

                # 获取该episode的元数据
                meta = episodes_meta.get(episode_index, {})
                tasks = meta.get('tasks', [])
                videos = meta.get('videos', {})

                # 构建新格式的record
                record = {
                    'state': data.get('observation.state', [])
                }

                # 添加视频信息，格式为 images_1, images_2, ...
                camera_list = sorted(videos.keys())  # 保证顺序一致
                for idx, camera_name in enumerate(camera_list, start=1):
                    video_info = videos[camera_name]
                    record[f'images_{idx}'] = {
                        'type': 'video',
                        'url': video_info['video_path'],
                        'frame_idx': frame_index
                    }

                # 添加prompt和is_robot
                record['prompt'] = tasks[0] if tasks else ""
                record['is_robot'] = True

                episode_data[episode_index].append(record)

    total_records = sum(len(records) for records in episode_data.values())
    print(f"✓ 读取了 {total_records} 条观察记录")
    print()

    # 4. 为每个episode创建一个jsonl文件
    print("生成episode文件...")
    for episode_index, records in tqdm(sorted(episode_data.items()), desc="创建文件"):
        output_file = output_dir / f"episode_{episode_index:04d}.jsonl"
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

    print(f"\n✓ 完成！共创建 {len(episode_data)} 个episode文件")
    print(f"✓ 输出目录: {output_dir}")

    return len(episode_data), total_records


def check_video_files(base_dir):
    """检查所有生成的jsonl文件中引用的视频文件是否存在"""
    print(f"\n{'='*80}")
    print(f"检查视频文件...")
    print(f"{'='*80}\n")

    base_path = Path(base_dir)
    missing_videos = []
    checked_count = 0
    missing_count = 0

    # 查找所有jsonl文件
    jsonl_files = sorted(base_path.rglob("main/jsonl/episode_*.jsonl"))

    if not jsonl_files:
        print("⚠️  未找到任何episode jsonl文件")
        return missing_videos

    print(f"✓ 找到 {len(jsonl_files)} 个episode文件，开始检查视频...")
    print()

    for jsonl_file in tqdm(jsonl_files, desc="检查视频文件"):
        # 获取数据集根目录：jsonl_file -> {dataset}/main/jsonl/episode_*.jsonl
        # 需要回溯到数据集根目录（即包含main文件夹的目录）
        # jsonl_file.parent = main/jsonl
        # jsonl_file.parent.parent = main
        # jsonl_file.parent.parent.parent = {dataset}
        dataset_root = jsonl_file.parent.parent.parent

        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # 检查所有images_*字段
                    for key in data.keys():
                        if key.startswith('images_'):
                            video_info = data[key]
                            if isinstance(video_info, dict) and 'url' in video_info:
                                video_rel_path = video_info['url']
                                # 视频路径格式: main/video/{camera_short_name}/observation.images.{camera_short_name}_episode_{episode_index:06d}.mp4
                                # 完整路径应该是: {dataset_root}/{video_rel_path}
                                video_full_path = dataset_root / video_rel_path
                                checked_count += 1

                                if not video_full_path.exists():
                                    missing_count += 1
                                    missing_videos.append({
                                        'dataset': dataset_root.name,
                                        'dataset_path': str(dataset_root),
                                        'episode_file': jsonl_file.name,
                                        'line_number': line_num,
                                        'video_rel_path': video_rel_path,
                                        'video_full_path': str(video_full_path),
                                        'key': key
                                    })
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON解析错误 {jsonl_file}:{line_num}: {e}")
                    continue

    print(f"\n✓ 检查完成！")
    print(f"  总检查数: {checked_count}")
    print(f"  缺失数: {missing_count}")

    return missing_videos


def save_missing_videos_report(missing_videos, output_dir):
    """保存缺失视频文件的报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not missing_videos:
        print(f"\n✓ 所有视频文件都存在！")
        return

    print(f"\n{'='*80}")
    print(f"保存缺失视频报告...")
    print(f"{'='*80}\n")

    # 保存详细的jsonl报告
    report_file = output_dir / "missing_videos.jsonl"
    with open(report_file, 'w') as f:
        for item in missing_videos:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ 详细报告已保存: {report_file}")
    print(f"  缺失视频总数: {len(missing_videos)}")

    # 按数据集统计
    dataset_stats = defaultdict(lambda: {'count': 0, 'episodes': set()})
    for item in missing_videos:
        dataset_name = item['dataset']
        episode_file = item['episode_file']
        dataset_stats[dataset_name]['count'] += 1
        dataset_stats[dataset_name]['episodes'].add(episode_file)

    # 保存统计报告
    summary_file = output_dir / "missing_videos_summary.json"
    summary_data = {}
    for dataset_name, stats in sorted(dataset_stats.items()):
        summary_data[dataset_name] = {
            'missing_count': stats['count'],
            'affected_episodes': sorted(list(stats['episodes']))
        }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 统计报告已保存: {summary_file}")

    # 打印统计信息
    print(f"\n缺失视频统计:")
    for dataset_name, stats in sorted(dataset_stats.items()):
        print(f"  {dataset_name}: {stats['count']} 个缺失视频")

    # 保存缺失视频路径列表（便于批量下载或处理）
    paths_file = output_dir / "missing_videos_paths.txt"
    with open(paths_file, 'w') as f:
        for item in missing_videos:
            f.write(f"{item['video_full_path']}\n")

    print(f"✓ 缺失视频路径列表已保存: {paths_file}")

    print(f"\n{'='*80}\n")


def batch_process_datasets(base_dir):
    """批量处理所有数据集"""
    print(f"{'='*80}")
    print(f"Episode JSONL 生成器 - 批量处理模式")
    print(f"{'='*80}")
    print(f"基础目录: {base_dir}")
    print(f"{'='*80}\n")

    # 查找所有数据集
    print("搜索数据集...")
    datasets = find_datasets(base_dir)

    if not datasets:
        print("⚠️  未找到任何数据集")
        return

    print(f"✓ 找到 {len(datasets)} 个数据集\n")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {Path(dataset).name}")
    print()

    # 处理每个数据集
    total_stats = {
        'datasets_processed': 0,
        'total_episodes': 0,
        'total_records': 0,
        'failed_datasets': []
    }

    for i, dataset_path in enumerate(datasets, 1):
        print(f"\n{'='*80}")
        print(f"处理数据集 [{i}/{len(datasets)}]")
        print(f"{'='*80}")

        try:
            num_episodes, num_records = process_single_dataset(dataset_path)

            if num_episodes > 0:
                total_stats['datasets_processed'] += 1
                total_stats['total_episodes'] += num_episodes
                total_stats['total_records'] += num_records
            else:
                total_stats['failed_datasets'].append(Path(dataset_path).name)

        except Exception as e:
            print(f"\n✗ 处理数据集失败: {str(e)}")
            import traceback
            traceback.print_exc()
            total_stats['failed_datasets'].append(Path(dataset_path).name)
            continue

    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量处理完成！")
    print(f"{'='*80}")
    print(f"总数据集数:     {len(datasets)}")
    print(f"成功处理:       {total_stats['datasets_processed']}")
    print(f"失败:           {len(total_stats['failed_datasets'])}")
    print(f"总episode数:    {total_stats['total_episodes']}")
    print(f"总记录数:       {total_stats['total_records']}")

    if total_stats['failed_datasets']:
        print(f"\n失败的数据集:")
        for dataset_name in total_stats['failed_datasets']:
            print(f"  - {dataset_name}")

    print(f"{'='*80}\n")

    # 检查视频文件
    missing_videos = check_video_files(base_dir)

    # 保存缺失视频报告
    report_dir = Path(base_dir) / "lapa"
    save_missing_videos_report(missing_videos, report_dir)


def main():
    parser = argparse.ArgumentParser(
        description="为每个episode创建jsonl文件，支持批量处理多个数据集"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/datasets/lerobot_2",
        help="基础目录，包含多个数据集"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="指定单个数据集路径（可选）"
    )

    args = parser.parse_args()

    if args.dataset:
        # 处理单个数据集
        process_single_dataset(args.dataset)
    else:
        # 批量处理所有数据集
        batch_process_datasets(args.base_dir)


if __name__ == "__main__":
    main()
