from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


import pyarrow.parquet as pq

# 读取 parquet 文件
table = pq.read_table('datasets/lerobot/aloha_mobile_cabinet/main/data/chunk-000/file-000.parquet')

# 转换为 pandas DataFrame
df = table.to_pandas()

print(df.head())