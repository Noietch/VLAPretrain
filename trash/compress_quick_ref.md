# 快速参考指南 - 多线程压缩

## 🚀 快速开始（3步）

### 1️⃣ 安装依赖
```bash
sudo apt-get update
sudo apt-get install -y p7zip-full pigz
```

### 2️⃣ 运行压缩（简单版）
```bash
chmod +x /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zip_file.sh
./zip_file.sh 8  # 使用8个线程
```

### 3️⃣ 或使用高级版
```bash
chmod +x /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/compress_advanced.sh

# 列出待压缩项
./compress_advanced.sh -l

# 开始压缩
./compress_advanced.sh -t 16 -m 7z -s 20g
```

---

## 📊 命令速查表

### 基础7z压缩（推荐）
```bash
# 最简单：压缩+自动分割（20GB）
7z a -v20g -mmt=8 -mx=9 output.7z /path/to/data

# 参数说明：
# a             = add（添加）
# -v20g        = 每个卷20GB（自动分割）
# -mmt=8       = 8个线程
# -mx=9        = 最高压缩（1-9）
# output.7z    = 输出文件
# /path/to/data = 源目录
```

### 解压7z分割文件
```bash
# 自动合并并解压
7z x output.7z.001 -o/target/path

# 验证完整性
7z t output.7z.001
```

### tar+pigz+split方案
```bash
# 方式1：压缩后分割
tar -cf - /data | pigz -9 -p 8 | pv -s 100G > data.tar.gz
split -b 20G data.tar.gz data.tar.gz.

# 方式2：解压
cat data.tar.gz.* | unpigz | tar -xf -

# 查看进度
pv -F '[%b %a, %r]' < input.tar.gz
```

### zip多线程（使用parallel）
```bash
# 如果要用zip的话
find /path/to/data -type f | \
  parallel -j 8 zip -q output.zip {}
```

---

## ⚡ 性能调优

### CPU充分利用
```bash
# 检查CPU核心数
nproc
# 通常设置线程 = 核心数

# 16核CPU示例
./compress_advanced.sh -t 16 -m 7z

# 查看实时CPU使用
htop
```

### 快速vs最优压缩
```bash
# 🚀 快速（3-5小时）
-mx=5  # 压缩级别改为5

# ⚖️ 平衡（6-12小时）
-mx=7  # 默认配置中的9改为7

# 🔒 最优（12-24小时）
-mx=9  # 最高压缩，默认
```

### 磁盘带宽优化
```bash
# 如果磁盘是瓶颈，减少线程数
./compress_advanced.sh -t 4  # 而不是8

# 监控磁盘I/O
iostat -x 1
```

---

## 📈 监控进度

### 实时查看文件大小
```bash
watch -n 2 'ls -lh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/*.7z*'
```

### 查看CPU/内存使用
```bash
top -p $(pgrep 7z)
```

### 查看磁盘使用
```bash
watch -n 5 'df -h /mnt/dolphinfs'
```

### 查看日志
```bash
tail -f /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/compress_*.log
```

---

## ✅ 验证压缩结果

### 检查文件完整性
```bash
# 7z验证
7z t ego4d_v2.7z.001

# 显示所有卷的文件列表
7z l ego4d_v2.7z.001 | head -20

# 统计压缩文件数量
7z l ego4d_v2.7z.001 | grep "files:" 
```

### 计算压缩率
```bash
# 原始大小
du -sh /mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2
# 如：500G

# 压缩后大小
du -sh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/
# 如：150G

# 压缩率 = 150/500 = 30%（节省70%）
```

---

## 🔧 问题排查

### 问题1：找不到命令
```bash
# 7z 不存在
sudo apt-get install p7zip-full

# pigz 不存在
sudo apt-get install pigz

# 验证安装
7z --version
pigz --version
```

### 问题2：权限不足
```bash
chmod +x compress_advanced.sh
chmod +x zip_file.sh
```

### 问题3：磁盘满
```bash
# 检查磁盘
df -h /mnt/dolphinfs

# 清理临时文件
rm -rf /tmp/*
rm -rf ~/.cache/*
```

### 问题4：压缩过程中断
```bash
# 7z 不支持断点续传，需要删除不完整文件并重新开始
rm /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/ego4d_v2.7z.*
./compress_advanced.sh
```

### 问题5：内存溢出
```bash
# 减少线程数或压缩级别
./compress_advanced.sh -t 4 -m 7z

# 在7z中改 -md=32m 为 -md=16m（字典大小）
```

---

## 🎯 最佳实践

### ✅ 推荐做法
```bash
# 1. 先检查空间
df -h /mnt/dolphinfs
du -sh /mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/*

# 2. 在后台运行（使用tmux/screen）
tmux new-session -d -s compress './compress_advanced.sh -t 16 -m 7z'

# 3. 查看进度
tmux attach -t compress

# 4. 定期验证
7z t ego4d_v2.7z.001

# 5. 保存压缩日志
cp /mnt/dolphinfs/.../compress_*.log ./backup/
```

### ❌ 避免做法
- ❌ 压缩系统盘（/） - 使用专用大容量盘
- ❌ 压缩时同时进行写操作 - 先停止其他I/O操作
- ❌ 压缩级别设为9的同时用过多线程 - 会大幅减速
- ❌ 不验证就删除原文件 - 先验证压缩完整性

---

## 📋 文件对照

| 文件 | 用途 | 使用场景 |
|------|------|--------|
| `zip_file.sh` | 简单版压缩脚本 | 新手，快速开始 |
| `compress_advanced.sh` | 高级版脚本 | 需要更多控制和监控 |
| `compress_guide.md` | 详细说明文档 | 深入学习各种方案 |
| `compress_quick_ref.md` | 本文档 | 快速查询命令 |

---

## 🔗 常用命令速查

```bash
# 📁 文件操作
ls -lh *.7z*                    # 列出压缩文件
du -sh /path                    # 查看目录大小
find . -name "*.7z*" -delete   # 删除所有7z文件

# 🔍 查询进度
watch -n 2 'ls -lh *.7z*'      # 实时看文件大小
htop -p $(pgrep 7z)           # CPU/内存使用
lsof | grep 7z                 # 7z打开的文件

# ✅ 验证
7z t output.7z.001            # 测试文件完整性
7z l output.7z.001 | wc -l    # 计算文件数
md5sum *.7z.* > checksums.md5 # 计算校验和

# 🔄 恢复/修复
7z repair output.7z.001       # 尝试修复（如果可能）

# 📊 统计
du -sb /data | awk '{printf "%.2f GB\n", $1/1e9}' # 显示GB单位
```

---

## 💡 根据硬件选择参数

### 8核CPU + 32GB内存
```bash
./compress_advanced.sh -t 8 -m 7z -s 20g
```

### 16核CPU + 64GB内存
```bash
./compress_advanced.sh -t 16 -m 7z -s 20g
```

### 32核CPU + 128GB内存
```bash
./compress_advanced.sh -t 32 -m 7z -s 20g
```

### 网络存储（NFS/HDFS）
```bash
# 减少线程以避免网络拥塞
./compress_advanced.sh -t 4 -m tar_pigz
```

---

需要帮助？查看详细文档：`compress_guide.md`
