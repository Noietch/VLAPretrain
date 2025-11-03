# 新的压缩脚本说明 - 分文件夹存储

## 📁 文件夹结构

更新后的脚本为每个压缩项创建单独的文件夹，文件夹名称与压缩项名称一致。

### 生成的目录结构

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/
├── ego4d_v2/
│   ├── ego4d_v2.7z.001  (20GB)
│   ├── ego4d_v2.7z.002  (20GB)
│   ├── ego4d_v2.7z.003  (20GB)
│   └── ... (其他分段)
├── lerobot/
│   ├── lerobot.7z.001   (20GB)
│   ├── lerobot.7z.002   (20GB)
│   └── ... (其他分段)
├── open-embodiment-X/
│   ├── open-embodiment-X.7z.001  (20GB)
│   ├── open-embodiment-X.7z.002  (20GB)
│   └── ... (其他分段)
└── ego4d/
    ├── ego4d.7z.001     (20GB)
    ├── ego4d.7z.002     (20GB)
    └── ... (其他分段)
```

## 🎯 优势

✅ **组织清晰**：每个数据集的压缩文件单独存放  
✅ **易于管理**：可以单独备份、删除或传输某个数据集  
✅ **便于追踪**：清楚地了解每个数据集的压缩进度  
✅ **易于解压**：知道哪些文件属于哪个数据集  
✅ **并行处理**：未来可支持同时压缩多个数据集

## 🚀 快速使用

### 运行简单版脚本
```bash
chmod +x zip_file.sh
./zip_file.sh 30  # 使用30个线程
```

输出示例：
```
========== 多线程压缩开始 ==========
开始压缩: ego4d_v2
输出文件: ego4d_v2
输出文件夹: /mnt/dolphinfs/.../zipped_file/ego4d_v2
线程数: 30
分割大小: 20GB
✓ 压缩完成: ego4d_v2
✓ 文件保存位置: /mnt/dolphinfs/.../zipped_file/ego4d_v2
...
========== 压缩完成 ==========
输出主目录: /mnt/dolphinfs/.../zipped_file

生成的文件夹结构:
drwxr-xr-x 2 hadoop-aipnlp hadoop-aipnlp 4.0K ego4d_v2
drwxr-xr-x 2 hadoop-aipnlp hadoop-aipnlp 4.0K ego4d
drwxr-xr-x 2 hadoop-aipnlp hadoop-aipnlp 4.0K lerobot
drwxr-xr-x 2 hadoop-aipnlp hadoop-aipnlp 4.0K open-embodiment-X

各文件夹内的分割文件:
ego4d_v2:
  /path/ego4d_v2/ego4d_v2.7z.001 (20G)
  /path/ego4d_v2/ego4d_v2.7z.002 (20G)
  /path/ego4d_v2/ego4d_v2.7z.003 (20G)
  ...
```

### 运行高级版脚本
```bash
chmod +x compress_advanced.sh

# 列出待压缩项
./compress_advanced.sh -l

# 压缩（使用16线程）
./compress_advanced.sh -t 16 -m 7z -s 20g

# 验证所有文件
./compress_advanced.sh -v
```

## 📊 文件操作示例

### 查看压缩进度
```bash
# 查看ego4d_v2的分割文件
ls -lh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/ego4d_v2/

# 实时监控所有文件大小
watch -n 2 'du -sh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/*'
```

### 解压特定数据集
```bash
# 解压ego4d_v2
7z x /mnt/dolphinfs/.../zipped_file/ego4d_v2/ego4d_v2.7z.001 -o/target/path

# 验证完整性
7z t /mnt/dolphinfs/.../zipped_file/ego4d_v2/ego4d_v2.7z.001

# 列出文件内容
7z l /mnt/dolphinfs/.../zipped_file/ego4d_v2/ego4d_v2.7z.001 | head -20
```

### 删除特定数据集
```bash
# 只删除ego4d的压缩文件
rm -rf /mnt/dolphinfs/.../zipped_file/ego4d/

# 保持其他数据集不受影响
```

### 传输特定数据集
```bash
# 将lerobot数据集复制到另一个位置
cp -r /mnt/dolphinfs/.../zipped_file/lerobot/ /backup/location/

# 或使用rsync进行远程传输
rsync -av /mnt/dolphinfs/.../zipped_file/lerobot/ remote-server:/backup/lerobot/
```

## 🔍 管理命令

### 获取各数据集的总大小
```bash
du -sh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/*/
```

输出示例：
```
120G  /path/zipped_file/ego4d_v2
80G   /path/zipped_file/lerobot
150G  /path/zipped_file/open-embodiment-X
100G  /path/zipped_file/ego4d
```

### 计算压缩率
```bash
# ego4d_v2原始大小 vs 压缩后大小
original_size=$(du -sb /mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2 | awk '{print $1}')
compressed_size=$(du -sb /mnt/dolphinfs/.../zipped_file/ego4d_v2 | awk '{print $1}')
ratio=$(echo "scale=2; ($compressed_size * 100) / $original_size" | bc)
echo "ego4d_v2 压缩率: ${ratio}%"
```

### 验证所有数据集
```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/

for dir in */; do
    echo "验证 $dir ..."
    if 7z t ${dir%/}/${dir%/}.7z.001 > /dev/null 2>&1; then
        echo "✓ $dir 验证通过"
    else
        echo "✗ $dir 验证失败"
    fi
done
```

## 🛠️ 高级用法

### 并行压缩（压缩不同的数据集）

```bash
#!/bin/bash
# parallel_compress.sh

# 在后台分别压缩不同的数据集
compress_advanced.sh -t 8 -m 7z -s 20g &
# 等待完成或在另一个终端启动另一个压缩任务
```

### 定时压缩（使用cron）

```bash
# 编辑crontab
crontab -e

# 添加以下行（每天晚上8点开始压缩）
0 20 * * * cd /path/to/LVLA && ./zip_file.sh 30 >> /var/log/compress.log 2>&1
```

### 监控压缩进度

```bash
#!/bin/bash
# monitor_compress.sh

while true; do
    clear
    echo "=== 压缩进度监控 ==="
    du -sh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/*/
    echo ""
    echo "总大小:"
    du -sh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/
    echo ""
    echo "CPU使用:"
    top -b -n 1 | grep "7z" | head -5
    sleep 10
done
```

## ✅ 检查清单

压缩完成后，确保：
- [ ] 每个数据集都有单独的文件夹
- [ ] 每个文件夹内的分割文件大小约为20GB（最后一个可能更小）
- [ ] 所有分割文件都存在且不损坏（使用 `7z t` 验证）
- [ ] 文件夹名称与数据集名称一致
- [ ] 没有遗漏任何数据集

## 📝 更新日志

### v2.1 (2025-11-03)
- ✅ 添加为每个压缩项创建单独文件夹的功能
- ✅ 改进输出显示，清晰展示文件夹结构
- ✅ 更新高级脚本以支持新的文件夹结构
- ✅ 优化验证和统计函数

## 🆘 常见问题

### Q: 如何只压缩某个数据集？
A: 编辑脚本，注释掉不需要压缩的行，然后运行。

### Q: 能否同时压缩多个数据集？
A: 可以，在不同的终端运行多个脚本实例，但要确保总线程数不超过CPU核心数。

### Q: 分割文件丢失了怎么办？
A: 无法恢复。确保在验证完整性后再删除原始数据。

### Q: 解压时需要所有分割文件吗？
A: 是的，7z需要所有分割文件来完成解压。缺少任何一个都会导致失败。

