# 更新总结 - 分文件夹压缩存储

## ✅ 已完成的更新

### 1. **zip_file.sh** (简单版)
- ✅ 为每个压缩项创建单独的文件夹
- ✅ 文件夹名称与压缩文件名一致
- ✅ 分割文件保存到对应的文件夹中
- ✅ 改进输出显示，列出所有文件夹结构

**使用方法：**
```bash
chmod +x zip_file.sh
./zip_file.sh 30  # 使用30个线程
```

### 2. **compress_advanced.sh** (高级版)
- ✅ 更新 `compress_7z()` 函数，为每个压缩项创建文件夹
- ✅ 更新 `compress_tar_pigz()` 函数，支持新的文件夹结构
- ✅ 改进 `verify_archives()` 函数，适配新的文件夹结构
- ✅ 改进 `show_stats()` 函数，以树形结构显示文件夹和文件

**使用方法：**
```bash
chmod +x compress_advanced.sh
./compress_advanced.sh -t 16 -m 7z -s 20g
```

### 3. **新增文档**
- ✅ `FOLDER_STRUCTURE_GUIDE.md` - 详细的文件夹结构说明和使用指南
- ✅ 包含管理命令、实际案例和故障排除

---

## 📁 新的目录结构

```
zipped_file/
├── ego4d_v2/              ← 单独的文件夹
│   ├── ego4d_v2.7z.001
│   ├── ego4d_v2.7z.002
│   ├── ego4d_v2.7z.003
│   └── ...
├── lerobot/               ← 单独的文件夹
│   ├── lerobot.7z.001
│   ├── lerobot.7z.002
│   └── ...
├── open-embodiment-X/     ← 单独的文件夹
│   ├── open-embodiment-X.7z.001
│   ├── open-embodiment-X.7z.002
│   └── ...
└── ego4d/                 ← 单独的文件夹
    ├── ego4d.7z.001
    ├── ego4d.7z.002
    └── ...
```

---

## 🎯 核心改进点

### 1️⃣ **组织清晰**
```bash
# 之前：所有文件混在一起
zipped_file/
├── ego4d_v2.7z.001
├── ego4d_v2.7z.002
├── lerobot.7z.001
├── lerobot.7z.002
└── ...

# 现在：按数据集分文件夹
zipped_file/
├── ego4d_v2/
│   ├── ego4d_v2.7z.001
│   └── ego4d_v2.7z.002
└── lerobot/
    ├── lerobot.7z.001
    └── lerobot.7z.002
```

### 2️⃣ **管理便捷**
```bash
# 查看特定数据集的大小
du -sh zipped_file/ego4d_v2/

# 只删除某个数据集
rm -rf zipped_file/ego4d_v2/

# 只备份某个数据集
cp -r zipped_file/lerobot/ /backup/
```

### 3️⃣ **便于追踪**
```bash
# 清楚地看到每个数据集的压缩状态
ls -lh zipped_file/*/

# 了解每个数据集有多少个分割文件
for dir in zipped_file/*/; do
    echo "$dir: $(ls -1 $dir/*.7z.* | wc -l) 个文件"
done
```

---

## 🚀 快速开始

### 方案 A：简单版（推荐）
```bash
cd /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA

chmod +x zip_file.sh
./zip_file.sh 30
```

### 方案 B：高级版（更多控制）
```bash
chmod +x compress_advanced.sh

# 查看待压缩项
./compress_advanced.sh -l

# 开始压缩
./compress_advanced.sh -t 16 -m 7z -s 20g

# 验证所有文件
./compress_advanced.sh -v
```

---

## 📊 脚本对比

| 功能 | zip_file.sh | compress_advanced.sh |
|------|---|---|
| 简单易用 | ✅ | ⭐⭐⭐ |
| 文件夹结构 | ✅ | ✅ |
| 参数自定义 | ❌ | ✅ |
| 日志记录 | ❌ | ✅ |
| 验证功能 | ❌ | ✅ |
| 进度监控 | 基础 | ✅ |

---

## 💾 文件位置

| 文件 | 位置 | 用途 |
|------|------|------|
| zip_file.sh | `/LVLA/zip_file.sh` | 简单压缩脚本 |
| compress_advanced.sh | `/LVLA/compress_advanced.sh` | 高级压缩脚本 |
| compress_guide.md | `/LVLA/compress_guide.md` | 详细方案对比 |
| compress_quick_ref.md | `/LVLA/compress_quick_ref.md` | 快速参考 |
| FOLDER_STRUCTURE_GUIDE.md | `/LVLA/FOLDER_STRUCTURE_GUIDE.md` | 文件夹结构说明（新增） |

---

## ✨ 关键特性

✅ **自动分文件夹** - 每个数据集单独存放  
✅ **20GB分割** - 每个分割文件约20GB  
✅ **多线程压缩** - 充分利用CPU  
✅ **清晰输出** - 显示完整的文件夹结构  
✅ **易于解压** - 所有分段自动合并  
✅ **便于管理** - 可单独操作各个数据集  

---

## 🔧 常用命令

### 监控压缩
```bash
watch -n 2 'du -sh /path/zipped_file/*/'
```

### 验证文件完整性
```bash
7z t /path/zipped_file/ego4d_v2/ego4d_v2.7z.001
```

### 解压
```bash
7z x /path/zipped_file/ego4d_v2/ego4d_v2.7z.001 -o/target/
```

### 查看文件列表
```bash
7z l /path/zipped_file/ego4d_v2/ego4d_v2.7z.001 | head -20
```

### 获取统计信息
```bash
du -sh /path/zipped_file/*/
```

---

## 📝 修改日志

### 2025-11-03
- ✅ 更新 `zip_file.sh`：添加单独文件夹功能
- ✅ 更新 `compress_advanced.sh`：所有函数适配新结构
- ✅ 新增 `FOLDER_STRUCTURE_GUIDE.md`：详细使用说明
- ✅ 改进输出显示和文件夹结构清晰度

---

## ❓ 常见问题

**Q：为什么要单独创建文件夹？**  
A：便于管理、追踪和单独操作各个数据集，特别是在处理大量数据时。

**Q：可以修改分割大小吗？**  
A：可以。在脚本中找到 `-v20g` 改为其他大小，如 `-v10g` 或 `-v50g`。

**Q：能否同时压缩多个数据集？**  
A：可以，但要注意CPU和磁盘负载。建议在不同终端运行多个脚本实例。

**Q：文件夹创建失败怎么办？**  
A：检查权限和磁盘空间。运行 `chmod 755 /path/zipped_file` 确保权限正确。

---

## 🎉 总结

现在您有了一套完整的多线程压缩解决方案，支持：
- 📁 **自动分文件夹存储** - 每个数据集单独管理
- ⚡ **多线程并行压缩** - 充分利用系统资源
- 💾 **20GB自动分割** - 便于存储和传输
- 🛠️ **灵活配置** - 支持自定义参数
- 📊 **完整监控** - 日志、验证、统计全齐全

立即开始使用：`./zip_file.sh 30`
