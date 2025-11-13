#!/bin/bash
# python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/data_pross/ocu_gpu.py &

# 源目录和目标目录
SOURCE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/datasets/lerobot_spec"
TARGET_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/datasets/pretrain/oxe"

# 需要提取的文件夹列表
FOLDERS_TO_EXTRACT=("jsonl" "latent_action" "video")

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 统计变量
total_datasets=0
processed_datasets=0
skipped_datasets=0

echo "=========================================="
echo "开始提取数据集文件夹"
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo "提取文件夹: ${FOLDERS_TO_EXTRACT[*]}"
echo "=========================================="
echo ""

# 遍历源目录中的所有数据集
for dataset in "$SOURCE_DIR"/*; do
    if [ -d "$dataset" ]; then
        dataset_name=$(basename "$dataset")
        total_datasets=$((total_datasets + 1))

        # 检查main文件夹是否存在
        main_dir="$dataset/main"
        if [ ! -d "$main_dir" ]; then
            echo "⚠️  跳过 $dataset_name: main文件夹不存在"
            skipped_datasets=$((skipped_datasets + 1))
            continue
        fi

        echo "📦 处理数据集: $dataset_name"

        # 创建目标数据集目录
        target_dataset_dir="$TARGET_DIR/$dataset_name/main"
        mkdir -p "$target_dataset_dir"

        # 提取每个指定的文件夹
        for folder in "${FOLDERS_TO_EXTRACT[@]}"; do
            source_folder="$main_dir/$folder"
            target_folder="$target_dataset_dir/$folder"

            if [ -d "$source_folder" ]; then
                echo "  ✓ 复制 $folder ..."
                # 使用cp进行递归复制，保留权限和时间戳
                cp -r -p -v "$source_folder" "$target_dataset_dir/" 2>&1 | tail -n 5

                if [ $? -eq 0 ]; then
                    echo "  ✅ $folder 复制完成"
                else
                    echo "  ❌ $folder 复制失败"
                fi
            else
                echo "  ⚠️  $folder 文件夹不存在，跳过"
            fi
        done

        processed_datasets=$((processed_datasets + 1))
        echo "  ✅ $dataset_name 处理完成"
        echo ""
    fi
done

echo "=========================================="
echo "提取完成！"
echo "总数据集数: $total_datasets"
echo "成功处理: $processed_datasets"
echo "跳过: $skipped_datasets"
echo "=========================================="
