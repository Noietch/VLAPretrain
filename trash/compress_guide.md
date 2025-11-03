# 多线程压缩指南（每部分20G）

## 方案对比

### 1. **推荐方案：7z（最佳）**
- ✅ **支持自动分割**：可以指定每个卷的大小（20GB）
- ✅ **多线程支持**：通过 `-mmt=8` 参数
- ✅ **高压缩率**：采用 LZMA2 算法
- ✅ **可靠**：广泛使用，开源稳定
- ❌ **解压需要7z工具**

**特性参数解释：**
```bash
-t7z              # 7z 格式
-m0=lzma2        # 压缩方法：LZMA2
-mx=9            # 压缩级别：9（最高）
-md=32m          # 字典大小：32MB
-v20g            # 分割大小：20GB（关键！）
-mmt=8           # 多线程：8个线程
-mfb=64          # 过滤器字节数
-ms=on           # Solid archive
```

**使用方法：**
```bash
./zip_file.sh 8  # 使用8个线程
```

**生成的文件示例：**
```
ego4d_v2.7z.001
ego4d_v2.7z.002
ego4d_v2.7z.003
...
```

**解压方法：**
```bash
# 自动解压所有部分
7z x ego4d_v2.7z.001

# 或指定输出目录
7z x ego4d_v2.7z.001 -o/path/to/output
```

---

### 2. **备选方案 A：tar + pigz（较快）**
- ✅ 多线程 gzip 压缩
- ✅ 广泛兼容性
- ❌ 不支持自动分割（需要手动分割）
- ⚠️ 压缩率不如7z

**使用方法：**
```bash
tar -cf - /path/to/data | pigz -9 -p 8 > output.tar.gz
```

**手动分割（使用split）：**
```bash
# 先压缩
tar -cf - /path/to/data | pigz -9 -p 8 > output.tar.gz

# 再分割（每20GB）
split -b 20G output.tar.gz output.tar.gz.
```

---

### 3. **备选方案 B：zip + parallel**
- ✅ 原生 zip 格式，兼容性最好
- ✅ 支持多线程处理
- ❌ zip 格式不支持跨部分压缩（分割后无法自动合并）
- ❌ 压缩率一般

---

## 安装依赖

### 安装 7z（推荐）
```bash
# Ubuntu/Debian
sudo apt-get install p7zip-full

# CentOS/RHEL
sudo yum install p7zip p7zip-plugins

# macOS
brew install p7zip
```

### 安装 pigz（备选）
```bash
# Ubuntu/Debian
sudo apt-get install pigz

# CentOS/RHEL
sudo yum install pigz

# macOS
brew install pigz
```

### 安装 GNU Parallel（可选）
```bash
# Ubuntu/Debian
sudo apt-get install parallel

# CentOS/RHEL
sudo yum install parallel
```

---

## 实际使用步骤

### 步骤 1：准备脚本
```bash
chmod +x /path/to/zip_file.sh
```

### 步骤 2：运行压缩
```bash
# 使用默认 8 线程
./zip_file.sh

# 或指定线程数
./zip_file.sh 16  # 使用 16 个线程
```

### 步骤 3：监控进度
```bash
# 查看文件大小变化
watch -n 2 'ls -lh /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/*.7z*'

# 或在另一个终端查看磁盘使用
watch -n 2 'df -h /mnt/dolphinfs'
```

### 步骤 4：验证完整性
```bash
# 测试 7z 文件完整性
7z t ego4d_v2.7z.001

# 或列出文件内容验证
7z l ego4d_v2.7z.001 | head -20
```

### 步骤 5：解压
```bash
# 方式 1：自动合并所有部分并解压
7z x ego4d_v2.7z.001 -o/output/path

# 方式 2：指定多个部分（手动）
7z x ego4d_v2.7z.001 ego4d_v2.7z.002 ego4d_v2.7z.003 ... -o/output/path
```

---

## 性能对比

| 方案 | 压缩速度 | 压缩率 | 多线程 | 自动分割 | 兼容性 |
|------|--------|------|------|--------|------|
| **7z** | 中等 | 最高 | ✅ | ✅ | ⭐⭐⭐ |
| tar+pigz | 快 | 中等 | ✅ | ❌ | ⭐⭐⭐⭐⭐ |
| tar+gzip | 较慢 | 中等 | ❌ | ❌ | ⭐⭐⭐⭐⭐ |
| zip | 快 | 低 | ❌ | ❌ | ⭐⭐⭐⭐⭐ |

---

## 故障排除

### 问题 1：磁盘空间不足
```bash
# 检查磁盘空间
df -h /mnt/dolphinfs

# 清理临时文件
rm -rf /tmp/*
```

### 问题 2：7z 命令不存在
```bash
# 安装 7z
sudo apt-get install p7zip-full

# 或使用 Docker
docker run -v /data:/data ubuntu:latest bash -c "apt-get update && apt-get install -y p7zip-full && 7z a -v20g ..."
```

### 问题 3：压缩速度慢
```bash
# 检查 CPU 使用率
top

# 增加线程数
./zip_file.sh 16

# 或调整压缩级别（在脚本中改 -mx=9 为 -mx=5）
```

### 问题 4：中断后恢复
7z 不支持断点续传。建议：
- 删除不完整的部分文件
- 重新运行压缩

---

## 脚本自定义

如需修改分割大小，编辑脚本中的以下行：
```bash
# 改为 10GB
-v10g

# 改为 50GB
-v50g
```

如需修改压缩级别（1=最快，9=最高压缩）：
```bash
# 快速压缩
-mx=5

# 最高压缩
-mx=9
```

---

## 总结

**推荐使用 7z 方案**，因为：
1. ✅ **自动分割**：无需手动处理分割逻辑
2. ✅ **多线程**：充分利用 CPU 资源
3. ✅ **压缩率高**：节省存储空间
4. ✅ **可靠稳定**：广泛使用，经过验证
5. ✅ **易于解压**：一条命令解压所有部分

现在运行脚本：
```bash
./zip_file.sh 8
```
