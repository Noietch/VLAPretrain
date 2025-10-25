#!/usr/bin/env python3
"""
脚本用于从Bridge数据集提取视频
"""

import sys
import os

# 添加数据加载器路径到Python路径
sys.path.append('./dataloder')

from bridge.dataset import BridgeDataset

def main():
    # 数据路径
    data_path = "/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024"

    # 输出视频保存路径
    output_dir = ".extract_data/bridge_video"
    output_video_path = os.path.join(output_dir, "bridge_video_sample.mp4")

    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_video_path}")

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return

    try:
        # 实例化BridgeDataset
        print("正在实例化BridgeDataset...")
        dataset = BridgeDataset(
            data_paths=[data_path],  # 数据路径列表
            seed=42,                 # 随机种子
            batch_size=1,           # 批次大小设为1便于提取视频
            train=False,            # 设为False避免shuffle和augmentation
            cache=False,            # 不缓存以节省内存
            load_language=False,    # 不加载语言标签
            skip_unlabeled=False,   # 不跳过未标记的数据
        )

        print("BridgeDataset实例化成功!")

        # 提取视频
        print("开始提取视频...")
        dataset.extracdeo(
            output_path=output_video_path,
            fps=5,              # 30帧每秒
            num_frames=300       # 提取300帧（约10秒视频）
        )

        print(f"视频提取完成! 保存在: {output_video_path}")

    except Exception as e:
        print(f"提取视频时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
