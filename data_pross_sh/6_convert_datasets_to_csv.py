#!/usr/bin/env python3
"""
将LeRobot数据集的info.json文件转换为CSV/Excel格式
每一行代表一个数据集
"""

import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any


def extract_dataset_info(info_path: Path, dataset_name: str) -> Dict[str, Any]:
    """从info.json文件中提取关键信息"""
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # 提取主要信息
        row = {
            'dataset_name': dataset_name,
            'codebase_version': info.get('codebase_version', ''),
            'robot_type': info.get('robot_type', ''),
            'total_episodes': info.get('total_episodes', 0),
            'total_frames': info.get('total_frames', 0),
            'total_tasks': info.get('total_tasks', 0),
            'chunks_size': info.get('chunks_size', 0),
            'fps': info.get('fps', 0),
        }

        # 提取特征信息
        features = info.get('features', {})

        # 统计图像特征数量
        image_features = [k for k in features.keys() if 'image' in k.lower()]
        row['num_image_features'] = len(image_features)

        # 列出所有图像特征名称
        row['image_features'] = ', '.join(image_features)

        # 提取action维度和格式
        if 'action' in features:
            action_shape = features['action'].get('shape', [])
            row['action_dim'] = action_shape[0] if action_shape else 0
            action_names = features['action'].get('names', {})
            if isinstance(action_names, dict) and 'motors' in action_names:
                row['action_format'] = ', '.join(action_names['motors'])
            else:
                row['action_format'] = ''

        # 提取observation.state维度和格式
        if 'observation.state' in features:
            state_shape = features['observation.state'].get('shape', [])
            row['state_dim'] = state_shape[0] if state_shape else 0
            state_names = features['observation.state'].get('names', {})
            if isinstance(state_names, dict) and 'motors' in state_names:
                row['state_format'] = ', '.join(state_names['motors'])
            else:
                row['state_format'] = ''

        # 提取observation.effort维度和格式
        if 'observation.effort' in features:
            effort_shape = features['observation.effort'].get('shape', [])
            row['effort_dim'] = effort_shape[0] if effort_shape else 0
            effort_names = features['observation.effort'].get('names', {})
            if isinstance(effort_names, dict) and 'motors' in effort_names:
                row['effort_format'] = ', '.join(effort_names['motors'])
            else:
                row['effort_format'] = ''

        return row

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return {
            'dataset_name': dataset_name,
            'error': str(e)
        }


def scan_datasets(base_path: str = 'datasets/lerobot') -> List[Dict[str, Any]]:
    """扫描所有数据集并提取信息"""
    base_dir = Path(base_path)

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    datasets_info = []

    # 遍历所有子目录
    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # 查找info.json文件（可能在main/meta/info.json或其他位置）
        info_paths = [
            dataset_dir / 'main' / 'meta' / 'info.json',
            dataset_dir / 'meta' / 'info.json',
            dataset_dir / 'info.json',
        ]

        info_path = None
        for path in info_paths:
            if path.exists():
                info_path = path
                break

        if info_path:
            print(f"Processing: {dataset_name}")
            dataset_info = extract_dataset_info(info_path, dataset_name)
            datasets_info.append(dataset_info)
        else:
            print(f"Warning: No info.json found for {dataset_name}")
            datasets_info.append({
                'dataset_name': dataset_name,
                'error': 'info.json not found'
            })

    return datasets_info


def main():
    """主函数"""
    print("Scanning datasets...")

    # 扫描所有数据集
    datasets_info = scan_datasets()

    if not datasets_info:
        print("No datasets found!")
        return

    # 转换为DataFrame
    df = pd.DataFrame(datasets_info)

    # 保存为CSV
    csv_path = 'datasets_info.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV file saved: {csv_path}")

    # 保存为Excel
    excel_path = 'datasets_info.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Excel file saved: {excel_path}")

    # 打印统计信息
    print(f"\nTotal datasets processed: {len(datasets_info)}")
    print(f"\nDatasets with errors: {df['error'].notna().sum() if 'error' in df.columns else 0}")

    # 显示前几行
    print("\nPreview of the data:")
    print(df.head(10).to_string())

    # 显示总结统计
    if 'total_episodes' in df.columns:
        print(f"\nTotal episodes across all datasets: {df['total_episodes'].sum()}")
    if 'total_frames' in df.columns:
        print(f"Total frames across all datasets: {df['total_frames'].sum()}")


if __name__ == '__main__':
    main()
