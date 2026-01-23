#!/usr/bin/env python3
"""
DDMTOLab 项目结构验证脚本
运行此脚本以验证项目是否准备好发布
"""

import os
import sys
from pathlib import Path


def check_file(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (未找到)")
        return False


def check_content(filepath, search_text, description):
    """检查文件内容"""
    if not os.path.exists(filepath):
        print(f"⚠️  {description}: {filepath} 不存在")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description}: 未找到 '{search_text}'")
                return False
    except Exception as e:
        print(f"❌ {description}: 读取错误 - {e}")
        return False


def main():
    print("=" * 60)
    print("DDMTOLab 项目结构验证")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    print("\n📁 项目结构检查:")
    print("-" * 60)

    # 核心结构
    checks = [
        ("src/ddmtolab/__init__.py", "包初始化文件"),
        ("src/ddmtolab/Algorithms", "Algorithms 目录"),
        ("src/ddmtolab/Problems", "Problems 目录"),
        ("src/ddmtolab/Methods", "Methods 目录"),
        ("README.md", "README 文件"),
        ("LICENSE", "LICENSE 文件"),
    ]

    for filepath, desc in checks:
        checks_total += 1
        if check_file(filepath, desc):
            checks_passed += 1

    print("\n📝 配置文件检查:")
    print("-" * 60)

    config_checks = [
        ("pyproject.toml", "PyPI 配置"),
        ("MANIFEST.in", "打包配置"),
        ("requirements.txt", "基础依赖"),
    ]

    for filepath, desc in config_checks:
        checks_total += 1
        if check_file(filepath, desc):
            checks_passed += 1

    print("\n🔍 内容验证:")
    print("-" * 60)

    # 检查版本号
    content_checks = [
        ("src/ddmtolab/__init__.py", "__version__", "版本号定义"),
        ("pyproject.toml", "version = ", "pyproject.toml 版本号"),
        ("pyproject.toml", 'package-dir = {"" = "src"}', "src 布局配置"),
    ]

    for filepath, search, desc in content_checks:
        checks_total += 1
        if check_content(filepath, search, desc):
            checks_passed += 1

    print("\n📦 可选文件:")
    print("-" * 60)

    optional_checks = [
        ("conda/meta.yaml", "Conda 配置"),
        ("requirements-dev.txt", "开发依赖"),
        ("environment.yml", "Conda 环境"),
        ("release.sh", "发布脚本"),
        (".gitignore", "Git 忽略文件"),
    ]

    for filepath, desc in optional_checks:
        if os.path.exists(filepath):
            print(f"✅ {desc}: {filepath}")
        else:
            print(f"⚪ {desc}: {filepath} (可选)")

    # 总结
    print("\n" + "=" * 60)
    print(f"检查结果: {checks_passed}/{checks_total} 通过")
    print("=" * 60)

    if checks_passed == checks_total:
        print("\n🎉 恭喜！项目结构完整，可以开始发布流程了！")
        print("\n下一步:")
        print("  1. 运行: pip install -e .")
        print("  2. 测试: python -c 'import ddmtolab; print(ddmtolab.__version__)'")
        print("  3. 构建: python -m build")
        print("  4. 发布: ./release.sh 或手动发布")
        return 0
    else:
        print("\n⚠️  还有一些文件缺失，请查看上面的清单。")
        print("\n建议:")
        print("  1. 将配置文件放置到项目根目录")
        print("  2. 确保 src/ddmtolab/__init__.py 包含版本信息")
        print("  3. 重新运行此脚本验证")
        return 1


if __name__ == "__main__":
    sys.exit(main())