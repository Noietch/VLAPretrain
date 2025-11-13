#!/usr/bin/env python3
"""
统计指定目录下所有 .jsonl 文件的个数
"""

import os
from pathlib import Path

def count_jsonl_files(directory):
    """
    递归统计目录下所有 .jsonl 文件的个数

    Args:
        directory: 要统计的目录路径

    Returns:
        dict: 包含总数和详细信息的字典
    """
    jsonl_files = []
    total_count = 0

    # 使用 Path 递归查找所有 .jsonl 文件
    for jsonl_file in Path(directory).rglob('*.jsonl'):
        jsonl_files.append(str(jsonl_file))
        total_count += 1

    return {
        'total_count': total_count,
        'files': sorted(jsonl_files)
    }

def main():
    # 目标目录
    target_dir = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/datasets/pretrain/oxe'

    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误: 目录不存在 - {target_dir}")
        return

    print(f"正在统计目录: {target_dir}")
    print("-" * 80)

    result = count_jsonl_files(target_dir)

    print(f"\n总共找到 {result['total_count']} 个 .jsonl 文件\n")

    if result['files']:
        print("文件列表:")
        for i, file in enumerate(result['files'], 1):
            print(f"{i:4d}. {file}")
    else:
        print("未找到任何 .jsonl 文件")

if __name__ == '__main__':
    main()
