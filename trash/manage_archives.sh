#!/bin/bash

##############################################
# 压缩文件管理工具
# 用于管理单独文件夹中的压缩文件
##############################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
ZIPPED_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file"

# 帮助函数
show_help() {
    cat << 'EOF'
压缩文件管理工具

用法: ./manage_archives.sh [命令] [选项]

命令:
  list              列出所有压缩数据集及其大小
  info <dataset>    显示指定数据集的详细信息
  verify <dataset>  验证指定数据集的完整性
  verify-all        验证所有数据集
  extract <dataset> <target>  解压指定数据集到目标位置
  size              显示各数据集的大小统计
  compare           对比原始数据和压缩数据的大小
  delete <dataset>  删除指定数据集的压缩文件（需确认）
  clean-temp        清理临时解压文件
  help              显示此帮助

示例:
  ./manage_archives.sh list
  ./manage_archives.sh info ego4d_v2
  ./manage_archives.sh verify ego4d_v2
  ./manage_archives.sh extract ego4d_v2 /tmp/
  ./manage_archives.sh size

EOF
}

# 列出所有数据集
list_datasets() {
    echo -e "${BLUE}========== 已压缩的数据集列表 ==========${NC}"
    echo ""
    
    if [ ! -d "$ZIPPED_DIR" ]; then
        echo -e "${RED}错误：压缩目录不存在 $ZIPPED_DIR${NC}"
        return 1
    fi
    
    local count=0
    for dir in "$ZIPPED_DIR"/*; do
        if [ -d "$dir" ]; then
            local dataset_name=$(basename "$dir")
            local file_count=$(ls -1 "$dir"/*.7z.* 2>/dev/null | wc -l)
            local total_size=$(du -sh "$dir" | awk '{print $1}')
            
            printf "%-30s  文件数: %2d  总大小: %8s\n" "$dataset_name" "$file_count" "$total_size"
            count=$((count + 1))
        fi
    done
    
    echo ""
    echo -e "${GREEN}总共 $count 个数据集${NC}"
}

# 显示数据集信息
show_info() {
    local dataset=$1
    local dataset_dir="$ZIPPED_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo -e "${RED}错误：数据集不存在: $dataset${NC}"
        return 1
    fi
    
    echo -e "${BLUE}========== $dataset 信息 ==========${NC}"
    echo ""
    
    echo -e "${CYAN}基本信息：${NC}"
    echo "  名称: $dataset"
    echo "  路径: $dataset_dir"
    echo "  总大小: $(du -sh "$dataset_dir" | awk '{print $1}')"
    echo ""
    
    echo -e "${CYAN}分割文件列表：${NC}"
    ls -lh "$dataset_dir"/*.7z.* 2>/dev/null | awk '{
        printf "  %-40s  %8s\n", $9, $5
    }'
    
    echo ""
    echo -e "${CYAN}文件数量：${NC}"
    local file_count=$(ls -1 "$dataset_dir"/*.7z.* 2>/dev/null | wc -l)
    echo "  $file_count 个分割文件"
    
    echo ""
    echo -e "${CYAN}预计解压大小（近似）：${NC}"
    # 显示7z内容信息
    if [ -f "$dataset_dir/$dataset.7z.001" ]; then
        local info=$(7z l "$dataset_dir/$dataset.7z.001" | tail -2 | head -1)
        echo "  $info"
    fi
}

# 验证单个数据集
verify_dataset() {
    local dataset=$1
    local dataset_dir="$ZIPPED_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo -e "${RED}错误：数据集不存在: $dataset${NC}"
        return 1
    fi
    
    echo -e "${BLUE}验证 $dataset...${NC}"
    
    if [ ! -f "$dataset_dir/$dataset.7z.001" ]; then
        echo -e "${RED}错误：找不到第一个分割文件${NC}"
        return 1
    fi
    
    if 7z t "$dataset_dir/$dataset.7z.001" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $dataset 验证通过${NC}"
        return 0
    else
        echo -e "${RED}✗ $dataset 验证失败${NC}"
        return 1
    fi
}

# 验证所有数据集
verify_all() {
    echo -e "${BLUE}========== 验证所有数据集 ==========${NC}"
    echo ""
    
    local passed=0
    local failed=0
    
    for dir in "$ZIPPED_DIR"/*; do
        if [ -d "$dir" ]; then
            local dataset_name=$(basename "$dir")
            if verify_dataset "$dataset_name"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        fi
    done
    
    echo ""
    echo -e "${GREEN}通过: $passed${NC}"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}失败: $failed${NC}"
    fi
}

# 解压数据集
extract_dataset() {
    local dataset=$1
    local target=$2
    local dataset_dir="$ZIPPED_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo -e "${RED}错误：数据集不存在: $dataset${NC}"
        return 1
    fi
    
    if [ -z "$target" ]; then
        echo -e "${RED}错误：请指定目标路径${NC}"
        return 1
    fi
    
    if [ ! -f "$dataset_dir/$dataset.7z.001" ]; then
        echo -e "${RED}错误：找不到第一个分割文件${NC}"
        return 1
    fi
    
    echo -e "${BLUE}解压 $dataset 到 $target...${NC}"
    echo ""
    
    mkdir -p "$target"
    7z x "$dataset_dir/$dataset.7z.001" -o"$target" -pmy2key
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 解压完成${NC}"
    else
        echo -e "${RED}✗ 解压失败${NC}"
        return 1
    fi
}

# 显示大小统计
show_size_stats() {
    echo -e "${BLUE}========== 大小统计 ==========${NC}"
    echo ""
    
    echo -e "${CYAN}各数据集大小：${NC}"
    du -sh "$ZIPPED_DIR"/* 2>/dev/null | sort -h | awk '{
        printf "  %-40s %8s\n", $2, $1
    }'
    
    echo ""
    echo -e "${CYAN}压缩目录总大小：${NC}"
    local total=$(du -sh "$ZIPPED_DIR" | awk '{print $1}')
    printf "  %s\n" "$total"
    
    echo ""
    echo -e "${CYAN}文件数量：${NC}"
    local file_count=$(find "$ZIPPED_DIR" -type f -name "*.7z.*" | wc -l)
    printf "  %d 个分割文件\n" "$file_count"
}

# 对比原始数据和压缩数据
compare_sizes() {
    echo -e "${BLUE}========== 原始 vs 压缩数据对比 ==========${NC}"
    echo ""
    
    local datasets=(
        "ego4d_v2|/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2"
        "lerobot|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/lerobot"
        "open-embodiment-X|/mnt/hdfs/user/hadoop-llm-data/dataset/open-embodiment-X/"
        "ego4d|/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d"
    )
    
    for item in "${datasets[@]}"; do
        IFS='|' read -r name path <<< "$item"
        
        if [ -e "$path" ]; then
            local original=$(du -sb "$path" 2>/dev/null | awk '{printf "%.2f", $1/1e9}')
            if [ -d "$ZIPPED_DIR/$name" ]; then
                local compressed=$(du -sb "$ZIPPED_DIR/$name" 2>/dev/null | awk '{printf "%.2f", $1/1e9}')
                local ratio=$(echo "scale=2; ($compressed * 100) / $original" | bc)
                
                printf "%-30s  原始: %8sGB  压缩: %8sGB  压缩率: %6s%%\n" \
                    "$name" "$original" "$compressed" "$ratio"
            else
                printf "%-30s  原始: %8sGB  压缩: %8s (未压缩)\n" "$name" "$original" "-"
            fi
        fi
    done
}

# 删除数据集
delete_dataset() {
    local dataset=$1
    local dataset_dir="$ZIPPED_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo -e "${RED}错误：数据集不存在: $dataset${NC}"
        return 1
    fi
    
    local size=$(du -sh "$dataset_dir" | awk '{print $1}')
    
    echo -e "${YELLOW}准备删除 $dataset (大小: $size)${NC}"
    echo ""
    echo "此操作不可恢复，请确认:"
    read -p "输入 'YES' 确认删除: " confirm
    
    if [ "$confirm" = "YES" ]; then
        rm -rf "$dataset_dir"
        echo -e "${GREEN}✓ 已删除 $dataset${NC}"
    else
        echo -e "${YELLOW}已取消删除${NC}"
    fi
}

# 清理临时文件
clean_temp() {
    echo -e "${BLUE}清理临时解压文件...${NC}"
    
    local temp_dirs=("/tmp/7z*" "$HOME/.7zTemp")
    
    for pattern in "${temp_dirs[@]}"; do
        if ls -d $pattern 2>/dev/null; then
            rm -rf $pattern
            echo -e "${GREEN}✓ 已删除: $pattern${NC}"
        fi
    done
    
    echo -e "${GREEN}清理完成${NC}"
}

# 主函数
main() {
    local command=${1:-"help"}
    
    case $command in
        list)
            list_datasets
            ;;
        info)
            show_info "$2"
            ;;
        verify)
            verify_dataset "$2"
            ;;
        verify-all)
            verify_all
            ;;
        extract)
            extract_dataset "$2" "$3"
            ;;
        size)
            show_size_stats
            ;;
        compare)
            compare_sizes
            ;;
        delete)
            delete_dataset "$2"
            ;;
        clean-temp)
            clean_temp
            ;;
        help)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $command${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 启动
main "$@"
