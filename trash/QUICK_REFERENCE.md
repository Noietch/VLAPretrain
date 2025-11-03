# 🎯 快速参考卡片 - 分文件夹压缩

## 📁 文件夹结构 (新增功能)

```
zipped_file/
├── ego4d_v2/          ← 单独文件夹
│   ├── ego4d_v2.7z.001 (20GB)
│   ├── ego4d_v2.7z.002 (20GB)
│   └── ...
├── lerobot/           ← 单独文件夹
├── open-embodiment-X/ ← 单独文件夹
└── ego4d/             ← 单独文件夹
```

---

## 🚀 三步快速开始

### 步骤1️⃣：准备脚本
```bash
cd /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA
chmod +x zip_file.sh
```

### 步骤2️⃣：开始压缩
```bash
# 使用30个线程
./zip_file.sh 30

# 或自定义线程数
./zip_file.sh 16
```

### 步骤3️⃣：查看结果
```bash
ls -lh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/zipped_file/
```

---

## 🛠️ 常用命令速查

### 管理工具命令（新增！）
```bash
# 列出所有数据集
./manage_archives.sh list

# 查看单个数据集信息
./manage_archives.sh info ego4d_v2

# 验证所有数据集
./manage_archives.sh verify-all

# 获取大小统计
./manage_archives.sh size

# 对比原始vs压缩大小
./manage_archives.sh compare

# 解压指定数据集
./manage_archives.sh extract ego4d_v2 /tmp/

# 删除指定数据集
./manage_archives.sh delete ego4d_v2
```

### 文件操作
```bash
# 查看特定数据集的文件
ls -lh zipped_file/ego4d_v2/

# 查看所有数据集大小
du -sh zipped_file/*/

# 实时监控
watch -n 2 'du -sh zipped_file/*/'

# 统计总大小
du -sh zipped_file/
```

### 验证和解压
```bash
# 验证ego4d_v2完整性
7z t zipped_file/ego4d_v2/ego4d_v2.7z.001

# 解压ego4d_v2
7z x zipped_file/ego4d_v2/ego4d_v2.7z.001 -o/target/path

# 查看文件列表
7z l zipped_file/ego4d_v2/ego4d_v2.7z.001
```

---

## 📊 可用脚本

| 脚本 | 复杂度 | 用途 | 命令 |
|------|------|------|------|
| `zip_file.sh` | ⭐ | 一键压缩 | `./zip_file.sh 30` |
| `compress_advanced.sh` | ⭐⭐⭐ | 高级配置 | `./compress_advanced.sh -t 16 -m 7z` |
| `manage_archives.sh` | ⭐⭐ | 管理工具 | `./manage_archives.sh list` |

---

## ✨ 主要特性

✅ **自动单独文件夹** - 每个数据集独立存放  
✅ **20GB分割** - 便于存储和传输  
✅ **多线程** - 充分利用CPU (可用30+线程)  
✅ **自动验证** - 验证完整性  
✅ **便捷管理** - 管理工具一应俱全  
✅ **易于解压** - 分段自动合并  

---

## 💡 实用技巧

### 监控压缩进度
```bash
# 方法1：查看文件大小变化
watch -n 2 'du -sh /path/zipped_file/*/'

# 方法2：查看CPU使用
top -p $(pgrep 7z)

# 方法3：查看磁盘I/O
iostat -x 1
```

### 后台运行压缩
```bash
# 使用tmux
tmux new-session -d -s compress './zip_file.sh 30'

# 查看进度
tmux attach -t compress

# 分离会话（Ctrl+B 再按D）
```

### 定时压缩
```bash
# 编辑crontab
crontab -e

# 每天晚上8点执行
0 20 * * * cd /path/LVLA && ./zip_file.sh 30 >> compress.log 2>&1
```

### 批量操作
```bash
# 验证所有数据集
for dir in zipped_file/*/; do
    echo "验证 $dir ..."
    7z t "${dir%/}/${dir%/}.7z.001" > /dev/null && echo "✓" || echo "✗"
done

# 计算所有数据集的压缩率
for dir in zipped_file/*/; do
    name=$(basename "$dir")
    size=$(du -sb "$dir" | awk '{print $1}')
    echo "$name: $(echo "scale=2; $size / 1e9" | bc) GB"
done
```

---

## 🔍 故障排查

| 问题 | 解决方案 |
|------|--------|
| `7zz: command not found` | `sudo apt-get install p7zip` |
| 权限不足 | `chmod +x *.sh` |
| 磁盘空间不足 | 检查 `df -h` |
| 压缩中断 | 删除不完整文件，重新运行 |
| 验证失败 | 检查磁盘、重新压缩 |

---

## 📈 性能参考

| 数据量 | 线程数 | 预计时间 | 压缩率 |
|------|------|--------|------|
| 500GB | 30 | 2-3小时 | 30-40% |
| 1TB | 30 | 4-6小时 | 30-40% |
| 2TB | 30 | 8-12小时 | 30-40% |

---

## 📁 文件位置

```
/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/
├── zip_file.sh                    ← 简单版脚本
├── compress_advanced.sh           ← 高级版脚本
├── manage_archives.sh             ← 管理工具（新增）
├── zipped_file/                   ← 输出文件夹
│   ├── ego4d_v2/
│   ├── lerobot/
│   ├── open-embodiment-X/
│   └── ego4d/
├── compress_guide.md              ← 详细指南
├── compress_quick_ref.md          ← 快速参考
├── FOLDER_STRUCTURE_GUIDE.md      ← 文件夹结构说明（新增）
├── UPDATE_SUMMARY.md              ← 更新总结（新增）
└── QUICK_REFERENCE.md             ← 本文件（新增）
```

---

## 🎯 常见场景

### 场景1：第一次压缩
```bash
./zip_file.sh 30
# 等待完成...
./manage_archives.sh verify-all
```

### 场景2：压缩特定数据集
编辑 `zip_file.sh`，注释掉不需要的行：
```bash
# 只压缩ego4d_v2
compress_with_7z "/mnt/hdfs/user/hadoop-aipnlp/aipnlpllm/dataset/video/ego4d_v2" "ego4d_v2"
# 其他行注释掉...
```

### 场景3：检查压缩进度
```bash
./manage_archives.sh size
./manage_archives.sh compare
```

### 场景4：解压并验证
```bash
./manage_archives.sh verify ego4d_v2
./manage_archives.sh extract ego4d_v2 /tmp/
```

### 场景5：备份特定数据集
```bash
cp -r zipped_file/ego4d_v2/ /backup/
# 或远程备份
rsync -av zipped_file/ego4d_v2/ remote-server:/backup/
```

---

## 🚀 立即开始

```bash
# 1. 进入目录
cd /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA

# 2. 准备脚本
chmod +x zip_file.sh manage_archives.sh

# 3. 开始压缩（使用30线程）
./zip_file.sh 30

# 4. 监控进度（另一个终端）
./manage_archives.sh size

# 5. 压缩完成后验证
./manage_archives.sh verify-all

# 6. 查看统计信息
./manage_archives.sh compare
```

---

## 📞 需要帮助？

查看详细文档：
- 📖 `compress_guide.md` - 完整技术文档
- 📋 `FOLDER_STRUCTURE_GUIDE.md` - 文件夹结构和管理
- 📝 `UPDATE_SUMMARY.md` - 本次更新说明

---

**最后更新：2025-11-03**  
**版本：2.1**  
**状态：✅ 已优化，可投入使用**
