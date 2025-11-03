#!/bin/bash

##############################################
# 高级多线程压缩脚本 v2.0
# 支持多种压缩方案、分割、断点恢复等
##############################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认配置
THREADS=${1:-8}
METHOD=${2:-"7z"}  # 压缩方法：7z, tar_pigz, zip
SPLIT_SIZE="20g"   # 分割大小
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets"
LOG_FILE="${OUTPUT_DIR}/compress_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 日志函数
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${CYAN}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[✓]${NC} ${timestamp} - ${message}"
            ;;
        WARN)
            echo -e "${YELLOW}[⚠]${NC} ${timestamp} - ${message}"
            ;;
        ERROR)
            echo -e "${RED}[✗]${NC} ${timestamp} - ${message}"
            ;;
    esac
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$LOG_FILE"
}

# 显示使用帮助
show_help() {
    cat << 'EOF'
使用方法: ./compress_advanced.sh [OPTIONS]

选项:
  -t, --threads NUM        线程数 (默认: 8)
  -m, --method METHOD      压缩方法 (默认: 7z)
                          可选: 7z, tar_pigz, zip
  -s, --split SIZE         分割大小 (默认: 20g)
                          可选: 10g, 20g, 50g 等
  -o, --output DIR         输出目录
  -l, --list              只显示待压缩项
  -v, --verify            验证压缩文件完整性
  -h, --help              显示此帮助

示例:
  # 8线程7z压缩，每20GB分割
  ./compress_advanced.sh -t 8 -m 7z -s 20g
  
  # 16线程tar+pigz压缩
  ./compress_advanced.sh -t 16 -m tar_pigz
  
  # 验证所有压缩文件
  ./compress_advanced.sh -v

EOF
}

# 检查依赖
check_dependencies() {
    local method=$1
    
    log INFO "检查依赖..."
    
    case $method in
        7z)
            if ! command -v 7z &> /dev/null; then
                log ERROR "7z 未安装"
                log INFO "请运行: sudo apt-get install p7zip-full"
                return 1
            fi
            log SUCCESS "7z 已安装: $(7z | head -1)"
            ;;
        tar_pigz)
            if ! command -v pigz &> /dev/null; then
                log ERROR "pigz 未安装"
                log INFO "请运行: sudo apt-get install pigz"
                return 1
            fi
            log SUCCESS "pigz 已安装: $(pigz --version)"
            ;;
        zip)
            if ! command -v zip &> /dev/null; then
                log ERROR "zip 未安装"
                return 1
            fi
            log SUCCESS "zip 已安装"
            ;;
    esac
    
    return 0
}

# 格式化文件大小
format_size() {
    local bytes=$1
    if [ "$bytes" -lt 1024 ]; then
        echo "${bytes}B"
    elif [ "$bytes" -lt 1048576 ]; then
        echo "$(( bytes / 1024 ))KB"
    elif [ "$bytes" -lt 1073741824 ]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        printf "%.2fGB\n" "$(echo "scale=2; $bytes / 1073741824" | bc)"
    fi
}

# 获取目录大小
get_dir_size() {
    local path=$1
    du -sb "$path" | awk '{print $1}'
}

# 列出待压缩项
list_items() {
    log INFO "待压缩项列表:"
    echo ""
    
    local items=(
        "/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2"
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/lerobot"
        "/mnt/hdfs/user/hadoop-llm-data/dataset/open-embodiment-X/"
        "/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d"
    )
    
    for i in "${!items[@]}"; do
        local path="${items[$i]}"
        if [ -e "$path" ]; then
            local size=$(get_dir_size "$path")
            printf "  [%d] %-40s %s\n" "$((i+1))" "$(basename "$path")" "$(format_size $size)"
        else
            printf "  [%d] %-40s ${RED}不存在${NC}\n" "$((i+1))" "$(basename "$path")"
        fi
    done
    
    echo ""
}

# 压缩函数 - 7z
compress_7z() {
    local source=$1
    local output_name=$2
    local source_name=$(basename "$source")
    
    # 为每个压缩项创建单独的文件夹
    local item_output_dir="$OUTPUT_DIR/${output_name}"
    mkdir -p "$item_output_dir"
    
    log INFO "开始 7z 压缩: $source_name"
    log INFO "参数: 线程=$THREADS, 分割大小=$SPLIT_SIZE"
    log INFO "输出文件夹: $item_output_dir"
    
    local start_time=$(date +%s)
    
    7z a -t7z \
        -m0=lzma2 \
        -mx=9 \
        -mfb=64 \
        -md=32m \
        -ms=on \
        -mhe=on \
        -v${SPLIT_SIZE} \
        -mmt=$THREADS \
        "$item_output_dir/${output_name}.7z" \
        "$source" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log SUCCESS "7z 压缩完成: $output_name (耗时 ${duration}s)"
        log INFO "文件保存位置: $item_output_dir"
        
        # 显示生成的文件
        local file_count=$(ls "$item_output_dir/${output_name}.7z"* 2>/dev/null | wc -l)
        log INFO "生成 $file_count 个分割文件"
    else
        log ERROR "7z 压缩失败: $output_name"
        return 1
    fi
}

# 压缩函数 - tar + pigz
compress_tar_pigz() {
    local source=$1
    local output_name=$2
    local source_name=$(basename "$source")
    
    # 为每个压缩项创建单独的文件夹
    local item_output_dir="$OUTPUT_DIR/${output_name}"
    mkdir -p "$item_output_dir"
    
    log INFO "开始 tar+pigz 压缩: $source_name"
    log INFO "参数: 线程=$THREADS"
    log INFO "输出文件夹: $item_output_dir"
    
    local start_time=$(date +%s)
    
    tar -cf - "$source" 2>/dev/null | pigz -9 -p $THREADS > "$item_output_dir/${output_name}.tar.gz"
    
    if [ $? -eq 0 ]; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log SUCCESS "tar+pigz 压缩完成: $output_name (耗时 ${duration}s)"
        log INFO "文件保存位置: $item_output_dir"
        
        local size=$(ls -lh "$item_output_dir/${output_name}.tar.gz" | awk '{print $5}')
        log INFO "输出文件大小: $size"
        
        # 如果需要分割
        log INFO "开始分割文件为 $SPLIT_SIZE 部分..."
        split -b $SPLIT_SIZE "$item_output_dir/${output_name}.tar.gz" "$item_output_dir/${output_name}.tar.gz."
        
        if [ $? -eq 0 ]; then
            log SUCCESS "文件分割完成"
            rm "$item_output_dir/${output_name}.tar.gz"  # 删除原文件
            log INFO "已删除原压缩文件，保留分割文件"
        fi
    else
        log ERROR "tar+pigz 压缩失败: $output_name"
        return 1
    fi
}

# 验证压缩文件
verify_archives() {
    log INFO "开始验证压缩文件..."
    
    case $METHOD in
        7z)
            log INFO "检查 7z 文件..."
            for dir in "$OUTPUT_DIR"/*; do
                if [ -d "$dir" ]; then
                    local dir_name=$(basename "$dir")
                    for file in "$dir"/${dir_name}.7z.001; do
                        if [ -f "$file" ]; then
                            log INFO "验证: $dir_name - $(basename "$file")"
                            if 7z t "$file" > /dev/null 2>&1; then
                                log SUCCESS "✓ $dir_name 验证通过"
                            else
                                log ERROR "✗ $dir_name 验证失败"
                            fi
                        fi
                    done
                fi
            done
            ;;
        tar_pigz)
            log INFO "检查 tar.gz 文件..."
            for dir in "$OUTPUT_DIR"/*; do
                if [ -d "$dir" ]; then
                    local dir_name=$(basename "$dir")
                    for file in "$dir"/${dir_name}.tar.gz.*; do
                        if [ -f "$file" ]; then
                            log INFO "验证: $dir_name - $(basename "$file")"
                            if tar -tzf "$file" > /dev/null 2>&1; then
                                log SUCCESS "✓ $dir_name 验证通过"
                            else
                                log ERROR "✗ $dir_name 验证失败"
                            fi
                        fi
                    done
                fi
            done
            ;;
    esac
}

# 显示统计信息
show_stats() {
    log INFO "压缩统计信息:"
    
    echo ""
    log INFO "文件夹结构:"
    ls -lhd "$OUTPUT_DIR"/* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    
    echo ""
    for dir in "$OUTPUT_DIR"/*; do
        if [ -d "$dir" ]; then
            local dir_name=$(basename "$dir")
            local dir_size=$(du -sh "$dir" | awk '{print $1}')
            echo -e "  ${CYAN}$dir_name:${NC} $dir_size"
            ls -lh "$dir"/* 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
        fi
    done
    
    echo ""
    local total_size=$(du -sh "$OUTPUT_DIR" | awk '{print $1}')
    log INFO "总大小: $total_size"
}

# 主程序
main() {
    log INFO "========== 高级压缩工具 v2.0 =========="
    log INFO "输出目录: $OUTPUT_DIR"
    log INFO "日志文件: $LOG_FILE"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--threads)
                THREADS=$2
                shift 2
                ;;
            -m|--method)
                METHOD=$2
                shift 2
                ;;
            -s|--split)
                SPLIT_SIZE=$2
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR=$2
                mkdir -p "$OUTPUT_DIR"
                shift 2
                ;;
            -l|--list)
                list_items
                exit 0
                ;;
            -v|--verify)
                verify_archives
                show_stats
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log ERROR "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查依赖
    if ! check_dependencies "$METHOD"; then
        exit 1
    fi
    
    log INFO "压缩方法: $METHOD, 线程数: $THREADS, 分割大小: $SPLIT_SIZE"
    
    # 定义压缩项
    local items=(
        "/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2|ego4d_v2"
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/lerobot|lerobot"
        "/mnt/hdfs/user/hadoop-llm-data/dataset/open-embodiment-X/|open-embodiment-X"
        "/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d|ego4d"
    )
    
    # 执行压缩
    local failed=0
    for item in "${items[@]}"; do
        IFS='|' read -r source output_name <<< "$item"
        
        if [ ! -e "$source" ]; then
            log WARN "路径不存在，跳过: $source"
            continue
        fi
        
        case $METHOD in
            7z)
                compress_7z "$source" "$output_name" || ((failed++))
                ;;
            tar_pigz)
                compress_tar_pigz "$source" "$output_name" || ((failed++))
                ;;
            *)
                log ERROR "未知压缩方法: $METHOD"
                exit 1
                ;;
        esac
    done
    
    echo ""
    log INFO "========== 压缩完成 =========="
    show_stats
    
    if [ $failed -gt 0 ]; then
        log ERROR "$failed 项压缩失败"
        exit 1
    else
        log SUCCESS "所有项压缩成功"
        exit 0
    fi
}

# 启动主程序
main "$@"
