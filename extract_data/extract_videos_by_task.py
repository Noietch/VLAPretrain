#!/usr/bin/env python3
import sys, os
import tensorflow as tf
import numpy as np
import cv2
import json
import glob
from pathlib import Path
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def decode_example(example_proto):
    # 根据实际的 TFRecord 格式定义特征描述
    feature_description = {
        'steps/observation/image_0': tf.io.VarLenFeature(tf.string),
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
    }

    try:
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        # 解析图像 (需要解码每个图像)
        images_encoded = tf.sparse.to_dense(parsed_features['steps/observation/image_0'])
        images = []
        for img_bytes in images_encoded.numpy():
            if img_bytes:
                img = tf.io.decode_image(img_bytes, channels=3).numpy()
                images.append(img)

        # 解析 actions (7维: xyz位置, xyz旋转, gripper)
        actions = tf.sparse.to_dense(parsed_features['steps/action']).numpy()
        # actions 是一维数组，需要 reshape 成 (num_steps, 7)
        if len(actions) > 0 and len(actions) % 7 == 0:
            num_steps = len(actions) // 7
            actions = actions.reshape(num_steps, 7)

        # 解析 language
        language_labels = tf.sparse.to_dense(parsed_features['steps/language_instruction'])

        return {
            'images': np.array(images) if images else np.array([]),
            'actions': actions,
            'language': language_labels.numpy()
        }
    except Exception as e:
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
        return {'images': np.array([]), 'actions': np.array([]), 'language': np.array([])}

def get_task_name(language):
    if isinstance(language, bytes):
        return language.decode('utf-8').strip()
    elif isinstance(language, str):
        return language.strip()
    elif isinstance(language, np.ndarray) and language.size > 0:
        lang_item = language.flat[0]
        return lang_item.decode('utf-8').strip() if isinstance(lang_item, bytes) else str(lang_item).strip()
    return None

def is_chunk_processed(data_path, output_dir):
    """
    检查 TFRecord 文件对应的 chunk 是否已经处理过

    Args:
        data_path: TFRecord 文件路径
        output_dir: 输出目录

    Returns:
        bool: 如果已处理返回 True，否则返回 False
    """
    import re
    match = re.search(r'-(\d+)-of-', os.path.basename(data_path))
    chunk_number = match.group(1) if match else "00000"
    chunk_dir_name = f"chunk-{chunk_number}"

    # 检查视频和 JSONL 目录是否存在且包含文件
    video_chunk_dir = os.path.join(output_dir, "video", chunk_dir_name)
    jsonl_chunk_dir = os.path.join(output_dir, "jsonl", chunk_dir_name)

    # 如果目录存在且包含文件，则认为已处理
    video_exists = os.path.exists(video_chunk_dir) and len(os.listdir(video_chunk_dir)) > 0
    jsonl_exists = os.path.exists(jsonl_chunk_dir) and len(os.listdir(jsonl_chunk_dir)) > 0

    return video_exists and jsonl_exists

def extract_videos_by_task(data_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)

    # 从文件名中提取编号，例如 "bridge_dataset-train.tfrecord-00001-of-01024" -> "00001"
    import re
    match = re.search(r'-(\d+)-of-', os.path.basename(data_path))
    chunk_number = match.group(1) if match else "00000"
    chunk_dir_name = f"chunk-{chunk_number}"

    # 创建视频和 JSONL 目录
    video_chunk_dir = os.path.join(output_dir, "video", chunk_dir_name)
    jsonl_chunk_dir = os.path.join(output_dir, "jsonl", chunk_dir_name)
    os.makedirs(video_chunk_dir, exist_ok=True)
    os.makedirs(jsonl_chunk_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(data_path)
    episode_count = 0

    for raw_record in dataset:
        try:
            parsed = decode_example(raw_record)
            images, actions, language = parsed["images"], parsed["actions"], parsed["language"]

            if len(images) == 0 or len(actions) == 0:
                continue

            task_name = get_task_name(language)
            if not task_name:
                continue

            episode_count += 1
            episode_name = f"episode_{episode_count:05d}"
            video_name = f"{episode_name}.mp4"
            jsonl_name = f"{episode_name}.jsonl"

            print(f"处理轨迹 {episode_count}: {task_name} (帧数: {len(images)}, 动作数: {len(actions)})")

            # 保存视频
            frame_height, frame_width = images[0].shape[:2]
            video_path = os.path.join(video_chunk_dir, video_name)
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            for frame in images:
                video_writer.write(frame)
            video_writer.release()

            # 为当前视频创建单独的 JSONL 文件
            jsonl_path = os.path.join(jsonl_chunk_dir, jsonl_name)
            with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                # 保存 actions 和 prompt 到 JSONL
                # 每个 action 是一个 7 维向量，每个 action 对应一个 frame
                for frame_idx, action in enumerate(actions):
                    json_entry = {
                        "state": action.tolist(),  # 转换为 Python list
                        "images_1": {
                            "type": "video",
                            "url": f"{chunk_dir_name}/{video_name}",
                            "frame_idx": frame_idx
                        },
                        "prompt": task_name,
                        "is_robot": True
                    }
                    jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"处理轨迹时出错: {e}")
            continue

    print(f"\n完成! 共提取 {episode_count} 个 episode")
    print(f"视频保存到: {video_chunk_dir}")
    print(f"JSONL 保存到: {jsonl_chunk_dir}")

def process_directory(directory_path, output_dir, fps=5):
    """
    处理文件夹内的所有 TFRecord 文件

    Args:
        directory_path: 包含 TFRecord 文件的文件夹路径
        output_dir: 输出目录
        fps: 视频帧率
    """
    # 检查路径是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 路径 '{directory_path}' 不是一个有效的文件夹")
        return

    # 查找所有 TFRecord 文件
    tfrecord_files = sorted(glob.glob(os.path.join(directory_path, "*.tfrecord*")))

    if not tfrecord_files:
        print(f"警告: 在 '{directory_path}' 中未找到任何 TFRecord 文件")
        return

    print(f"找到 {len(tfrecord_files)} 个 TFRecord 文件")

    # 过滤掉已处理的文件
    files_to_process = []
    skipped_files = []

    for tfrecord_file in tfrecord_files:
        if is_chunk_processed(tfrecord_file, output_dir):
            skipped_files.append(tfrecord_file)
        else:
            files_to_process.append(tfrecord_file)

    print(f"已处理文件: {len(skipped_files)} 个")
    print(f"待处理文件: {len(files_to_process)} 个")

    if len(files_to_process) == 0:
        print("\n所有文件都已处理完成，无需重新处理！")
        return

    print("=" * 80)

    # 处理每个文件
    for idx, tfrecord_file in enumerate(files_to_process, 1):
        print(f"\n[{idx}/{len(files_to_process)}] 处理文件: {os.path.basename(tfrecord_file)}")
        print("-" * 80)
        try:
            extract_videos_by_task(tfrecord_file, output_dir, fps)
        except Exception as e:
            print(f"处理文件 '{tfrecord_file}' 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print(f"所有文件处理完成! (处理: {len(files_to_process)}, 跳过: {len(skipped_files)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="提取 TFRecord 文件中的视频和动作数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理文件夹内的所有文件
  python extract_videos_by_task.py --input /path/to/tfrecord/folder --output ./extract_data --fps 5

  # 处理单个文件
  python extract_videos_by_task.py --input /path/to/file.tfrecord --output ./extract_data --fps 5 --single
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入路径：可以是文件夹（处理所有 TFRecord 文件）或单个 TFRecord 文件'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./extract_data',
        help='输出目录路径（默认: ./extract_data）'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='视频帧率（默认: 5）'
    )

    parser.add_argument(
        '--single',
        action='store_true',
        help='如果指定此标志，将输入路径视为单个文件而不是文件夹'
    )

    args = parser.parse_args()

    # 验证输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径 '{args.input}' 不存在")
        sys.exit(1)

    # 根据参数选择处理方式
    if args.single or os.path.isfile(args.input):
        # 处理单个文件
        if not os.path.isfile(args.input):
            print(f"错误: '{args.input}' 不是一个有效的文件")
            sys.exit(1)
        print(f"处理单个文件: {args.input}")
        extract_videos_by_task(args.input, args.output, fps=args.fps)
    else:
        # 处理文件夹内的所有文件
        print(f"处理文件夹: {args.input}")
        process_directory(args.input, args.output, fps=args.fps)
