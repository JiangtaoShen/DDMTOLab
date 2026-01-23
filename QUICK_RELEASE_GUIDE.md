# DDMTOLab 版本更新快速指南

## 🚀 每次更新只需 5 步！

---

## 步骤 1: 更新版本号 (2 分钟)

打开并修改这 **4 个文件**：

### 文件 1: `src\ddmtolab\__init__.py`
```python
__version__ = "1.1.0"  # ← 改这里
```

### 文件 2: `pyproject.toml`
```toml
version = "1.1.0"  # ← 改这里（第 7 行左右）
```

### 文件 3: `conda\meta.yaml`
```yaml
{% set version = "1.1.0" %}  # ← 改这里（第 2 行）
```

### 文件 4: `release.bat`
```batch
set VERSION=1.1.0  # ← 改这里（第 7 行）
```

💡 **提示**：可以用 Ctrl+F 搜索 "1.0.0" 快速定位

---

## 步骤 2: 更新 CHANGELOG (1 分钟)

编辑 `CHANGELOG.md`（如果没有就创建）：

```markdown
# Changelog

## [1.1.0] - 2025-01-26

### Added
- 新增 XX 算法
- 新增 YY 功能

### Fixed
- 修复 ZZ bug

## [1.0.0] - 2025-01-24
- 初始发布
```

---

## 步骤 3: 提交到 GitHub (1 分钟)

打开 **Anaconda Prompt** 或 **命令提示符**：

```bash
# 进入项目目录
cd D:\DDMTOLab

# 提交所有修改
git add .
git commit -m "release: bump version to 1.1.0"
git push origin main
```

---

## 步骤 4: 运行发布脚本 (3 分钟)

### 方法 A: 双击运行（最简单）
1. 双击 `D:\DDMTOLab\release.bat`
2. 按提示操作（一路回答 y 或 n）

### 方法 B: 命令行运行
```bash
cd D:\DDMTOLab
release.bat
```

### 脚本会自动执行：
1. ✅ 清理旧的构建文件
2. ✅ 构建新包
3. ✅ 检查包完整性
4. ✅ 询问是否上传到 TestPyPI（可选）
5. ✅ 上传到 PyPI
6. ✅ 创建 Git 标签

### 脚本提示说明：

```
Upload to TestPyPI for testing? (y/n):
```
- 输入 `n` 跳过测试（常规更新）
- 输入 `y` 先测试再发布（重大更新推荐）

```
Upload to official PyPI? (y/n):
```
- 输入 `y` 发布到 PyPI（必需）

```
Create and push Git tag? (y/n):
```
- 输入 `y` 创建版本标签（推荐）

---

## 步骤 5: 创建 GitHub Release (2 分钟)

脚本运行完后会提示：
```
Please create a GitHub Release at:
https://github.com/JiangtaoShen/DDMTOLab/releases/new
```

### 操作：
1. **点击链接**打开 GitHub
2. **选择标签**: `v1.1.0`
3. **Release 标题**: `v1.1.0 - 简短描述`
4. **描述**（复制模板）:

```markdown
## What's New

### 🚀 New Features
- [列出新功能]

### 🐛 Bug Fixes
- [列出修复的问题]

### 📈 Improvements
- [列出改进]

### 📦 Installation
```bash
pip install --upgrade ddmtolab
```

**Full Changelog**: https://github.com/JiangtaoShen/DDMTOLab/compare/v1.0.0...v1.1.0
```

5. **点击 "Publish release"**

---

## ✅ 完成！验证发布

等待 2-3 分钟，然后验证：

```bash
# 在新环境测试
conda create -n test_new python=3.10
conda activate test_new
pip install ddmtolab
python -c "import ddmtolab; print(ddmtolab.__version__)"
```

应该显示：`1.1.0` ✅

---

## 📋 快速检查清单

发布前确认：
- [ ] 4 个文件的版本号已更新
- [ ] CHANGELOG.md 已更新
- [ ] 代码已测试通过
- [ ] Git 已提交并推送

发布流程：
- [ ] 运行 `release.bat`
- [ ] 上传到 PyPI
- [ ] 创建 Git 标签
- [ ] 创建 GitHub Release
- [ ] 验证新版本可安装

---

## 🎯 常见情况

### 情况 1: 小 bug 修复
```
版本: 1.0.0 → 1.0.1
改动: 只改第 3 个数字
```

### 情况 2: 新增功能
```
版本: 1.0.1 → 1.1.0
改动: 改第 2 个数字，第 3 个归零
```

### 情况 3: 重大更新
```
版本: 1.1.0 → 2.0.0
改动: 改第 1 个数字，后两个归零
```

---

## ⚡ 超快速版（熟练后）

```bash
# 1. 改版本号（4个文件）
# 2. 更新 CHANGELOG
# 3. Git 提交
git add . && git commit -m "release: v1.1.0" && git push

# 4. 发布
release.bat

# 5. GitHub Release（网页操作）
```

**总耗时：5-10 分钟** 🚀

---

## 🆘 出问题了怎么办？

### 问题 1: release.bat 运行失败
```bash
# 检查是否在正确目录
cd D:\DDMTOLab

# 检查是否激活环境
conda activate ddmtolab

# 重新运行
release.bat
```

### 问题 2: twine 上传失败
```bash
# 检查 .pypirc 配置
type C:\Users\SJT\.pypirc

# 确认 API token 正确
# 重新上传
twine upload dist/*
```

### 问题 3: 忘记更新某个文件的版本号
**不能修改已发布的版本！**
```bash
# 只能发布新版本（如 1.1.1）
# 更新遗漏的文件
# 重新发布
```

---

## 📌 保存这份文档

把这个文件保存为 `RELEASE.md` 放在项目根目录，每次更新时打开查看！

---

**🎉 就这么简单！5 个步骤，10 分钟搞定版本发布！**
