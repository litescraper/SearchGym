#!/usr/bin/env python3
"""
简化的统一执行脚本 - 使用统一配置运行所有步骤
"""
import sys
import os
import subprocess
import argparse
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.unified_builder import UNIFIED_CONFIG, get_name_generation_config, get_path_generation_config, get_complex_paths_config, get_file_paths_config, get_qa_generation_config, get_simple_qa_templates_config

def run_script(script_name, description, force_run=False):
    """运行指定脚本"""
    print(f"=== {description} ===")
    print(f"运行脚本: {script_name}")

    # 特殊处理名称生成脚本：检查各个类别的名称文件是否都存在
    if script_name == 'scripts/generate_names.py' and not force_run:
        from scripts.unified_builder import check_names_files_exist

        all_files_exist, missing_files, existing_files = check_names_files_exist()

        if all_files_exist:
            print("✓ 所有名称文件都已存在，跳过名称生成步骤")
            print(f"  已存在的文件: {', '.join(existing_files)}")
            print()
            return True
        else:
            print(f"发现缺失的名称文件: {', '.join(missing_files)}")
            print(f"已存在的文件: {', '.join(existing_files) if existing_files else '无'}")
            print("将重新运行名称生成步骤以补充缺失的文件...")
            print()

            # 如果只有部分文件缺失，修改generate_names.py的参数只生成缺失的类型
            if missing_files and existing_files:
                print(f"注意：只为缺失的类型生成名称: {', '.join(missing_files)}")
                # 修改脚本调用参数，只生成缺失的类型
                result = subprocess.run([
                    sys.executable, script_name, "--type"] + missing_files,
                    text=True, timeout=3600)

                if result.returncode == 0:
                    print(f"✓ {description} 完成 (耗时: {datetime.now() - start_time})")
                else:
                    print(f"✗ {description} 失败")
                    return False
                print()
                return True

    start_time = datetime.now()

    try:
        result = subprocess.run([
            sys.executable, script_name
        ], text=True, timeout=3600)  # 1小时超时，不捕获输出以显示进度条

        if result.returncode == 0:
            print(f"✓ {description} 完成 (耗时: {datetime.now() - start_time})")
        else:
            print(f"✗ {description} 失败")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {description} 超时")
        return False
    except Exception as e:
        print(f"✗ {description} 出现异常: {e}")
        return False

    print()
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用统一配置运行所有知识图谱构建步骤')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['names', 'nodes', 'kg', 'paths', 'complex_paths', 'simple_qa', 'complex_qa', 'all'],
                       default=['all'],
                       help='指定要执行的步骤')
    parser.add_argument('--show-config', action='store_true',
                       help='显示当前配置')
    parser.add_argument('--force-names', action='store_true',
                       help='强制重新生成名称（即使已存在）')

    args = parser.parse_args()

    if args.show_config:
        print("="*60)
        print("统一配置")
        print("="*60)
        from scripts.unified_builder import print_config
        print_config()
        return

    print("="*60)
    print("统一知识图谱构建流程")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    print()

    # 创建所有必要的目录（除了names目录，因为需要特殊检查）
    for dir_path in get_file_paths_config().values():
        if isinstance(dir_path, str) and dir_path.endswith('/') and dir_path != 'outputs/names/':
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
    print()

    # 步骤映射
    step_scripts = {
        'names': 'scripts/generate_names.py',
        'nodes': 'scripts/build_nodes.py',
        'kg': 'scripts/build_kg.py',
        'paths': 'scripts/build_paths.py',
        'complex_paths': 'scripts/build_complex_paths.py',
        'simple_qa': 'scripts/generate_simple_qa.py',
        'complex_qa': 'scripts/generate_complex_qa.py'
    }

    step_descriptions = {
        'names': '步骤1: 生成名称',
        'nodes': '步骤2: 构建节点',
        'kg': '步骤3: 构建知识图谱',
        'paths': '步骤4: 构建路径',
        'complex_paths': '步骤5: 构建复杂路径',
        'simple_qa': '步骤6: 生成简单QA',
        'complex_qa': '步骤7: 生成复杂QA'
    }

    # 获取QA生成配置
    qa_config = get_qa_generation_config()
    generate_path = qa_config["generate_path"]
    generate_qa = qa_config["generate_qa"]

    # 显示配置信息
    print("当前配置摘要:")
    print(f"  名称生成: {get_name_generation_config()}")
    print(f"  路径生成: {get_path_generation_config()}")
    print(f"  复杂路径: {get_complex_paths_config()}")
    print(f"  QA生成: generate_path={generate_path}, generate_qa={generate_qa}")
    print()

    # 根据配置确定要执行的步骤
    if 'all' in args.steps:
        if generate_path and not generate_qa:
            # 场景1: 只生成路径
            steps_to_run = ['names', 'nodes', 'kg', 'paths', 'complex_paths']
            print("执行模式: 只生成路径（不生成QA）")
        elif not generate_path and generate_qa:
            # 场景2: 只生成QA
            # 检查路径文件是否存在
            from scripts.unified_builder import check_path_files_exist
            path_files_exist, missing_files = check_path_files_exist()
            if not path_files_exist:
                print("错误: 缺少必需的路径文件，请先运行 generate_path=True 来生成路径:")
                for file_path in missing_files:
                    print(f"  - {file_path}")
                return
            print("✓ 所有必需的路径文件都存在，开始生成QA...")
            steps_to_run = ['simple_qa', 'complex_qa']
            print("执行模式: 只生成QA（基于已有路径文件）")
        elif generate_path and generate_qa:
            # 场景3: 生成所有内容
            steps_to_run = ['names', 'nodes', 'kg', 'paths', 'complex_paths', 'simple_qa', 'complex_qa']
            print("执行模式: 生成路径和QA")
        else:
            print("错误: 无效的配置组合")
            return
    else:
        # 用户指定了具体步骤，按用户指定的步骤执行
        steps_to_run = args.steps
        print(f"执行用户指定的步骤: {steps_to_run}")

    print()

    # 执行步骤
    for step in steps_to_run:
        if step in step_scripts:
            force_run = args.force_names if step == 'names' else False
            run_script(step_scripts[step], step_descriptions[step], force_run)
        else:
            print(f"警告: 未知步骤 '{step}'")

    print("="*60)
    print(f"所有步骤执行完成! 结束时间: {datetime.now()}")
    print("="*60)

if __name__ == "__main__":
    main()