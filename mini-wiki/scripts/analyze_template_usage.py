#!/usr/bin/env python3
"""
分析forward_qa.jsonl中模板使用情况的脚本
统计各个hop对应的template占整个此hop所有问题的比例
并与UNIFIED_CONFIG中的配置进行比较
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict, Counter
# from scripts.unified_builder import get_simple_qa_templates_config

def get_simple_qa_templates_config():
    """
    直接从文件读取模板配置，避免依赖问题
    """
    # 根据之前添加的配置，直接在这里定义
    return {
        "1-hop": {
            "simple_qa_prompt_all_template": 0.3,
            "simple_qa_prompt_scenario_1_hop_template": 0.3,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "2-hop": {
            "simple_qa_prompt_all_template": 0.3,
            "simple_qa_prompt_scenario_2_hop_template": 0.3,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "3-hop": {
            "simple_qa_prompt_all_template": 0.3,
            "simple_qa_prompt_scenario_3_hop_template": 0.3,
            "simple_qa_prompt_123_hop_template": 0.4
        },
        "4-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        },
        "5-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        },
        "6-hop": {
            "simple_qa_prompt_all_template": 0.5,
            "simple_qa_prompt_456_hop_template": 0.5
        }
    }

def analyze_template_usage(qa_file_path):
    """
    分析QA文件中的模板使用情况
    """
    # 读取QA文件
    qa_data = []
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_entry = json.loads(line.strip())
                    qa_data.append(qa_entry)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {qa_file_path}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"错误: 文件格式不正确 {qa_file_path}: {e}")
        return None, None

    print(f"读取了 {len(qa_data)} 条QA记录")

    # 按hop分组统计模板使用情况
    actual_usage = defaultdict(lambda: defaultdict(int))
    hop_counts = defaultdict(int)

    for qa_entry in qa_data:
        path_length = qa_entry.get('path_length', 0)
        template_used = qa_entry.get('template_used', 'unknown')

        if path_length > 0:  # 只统计有效的hop
            hop_key = f"{path_length}-hop"
            actual_usage[hop_key][template_used] += 1
            hop_counts[hop_key] += 1

    # 计算实际使用比例
    actual_percentages = {}
    for hop_key, templates in actual_usage.items():
        total = hop_counts[hop_key]
        actual_percentages[hop_key] = {
            template: count / total * 100
            for template, count in templates.items()
        }

    return actual_usage, actual_percentages

def get_config_percentages():
    """
    获取配置中的模板比例（转换为百分比）
    """
    config = get_simple_qa_templates_config()
    config_percentages = {}

    for hop_key, templates in config.items():
        config_percentages[hop_key] = {
            template: weight * 100
            for template, weight in templates.items()
        }

    return config_percentages

def compare_actual_vs_config(actual_percentages, config_percentages):
    """
    比较实际使用情况和配置的差异
    """
    print("\n" + "="*80)
    print("模板使用情况分析报告")
    print("="*80)

    all_hops = set(actual_percentages.keys()) | set(config_percentages.keys())

    for hop_key in sorted(all_hops, key=lambda x: int(x.split('-')[0])):
        print(f"\n{hop_key}:")
        print("-" * 60)

        actual = actual_percentages.get(hop_key, {})
        config = config_percentages.get(hop_key, {})

        # 获取所有模板名称
        all_templates = set(actual.keys()) | set(config.keys())

        if not all_templates:
            print("  无数据")
            continue

        print(f"{'模板名称':<40} {'实际比例':<12} {'配置比例':<12} {'差异':<10}")
        print("-" * 70)

        total_actual = sum(actual.values())
        total_config = sum(config.values())

        for template in sorted(all_templates):
            actual_pct = actual.get(template, 0)
            config_pct = config.get(template, 0)
            diff = actual_pct - config_pct

            print(f"{template:<40} {actual_pct:>10.2f}% {config_pct:>10.2f}% {diff:>+8.2f}%")

        # 汇总信息
        print("-" * 70)
        print(f"{'汇总':<40} {total_actual:>10.2f}% {total_config:>10.2f}%")

        # 计算差异统计
        if config:
            max_diff = max(abs(actual.get(t, 0) - config.get(t, 0)) for t in all_templates)
            avg_diff = sum(abs(actual.get(t, 0) - config.get(t, 0)) for t in all_templates) / len(all_templates)
            print(f"\n最大差异: {max_diff:.2f}%")
            print(f"平均差异: {avg_diff:.2f}%")

def print_summary_statistics(actual_usage, actual_percentages):
    """
    打印汇总统计信息
    """
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)

    total_questions = 0
    for hop_key, templates in actual_usage.items():
        hop_total = sum(templates.values())
        total_questions += hop_total
        print(f"{hop_key}: {hop_total} 个问题")

    print(f"\n总问题数: {total_questions}")

    # 模板使用多样性统计
    print("\n模板使用多样性:")
    for hop_key, templates in actual_usage.items():
        print(f"{hop_key}: 使用了 {len(templates)} 种不同模板")

def main():
    """
    主函数
    """
    qa_file_path = 'outputs/qa/forward_qa.jsonl'

    print("开始分析模板使用情况...")
    print(f"QA文件路径: {qa_file_path}")

    # 分析实际使用情况
    actual_usage, actual_percentages = analyze_template_usage(qa_file_path)
    if actual_usage is None:
        return

    # 获取配置比例
    config_percentages = get_config_percentages()

    # 打印配置信息
    print("\n配置的模板比例:")
    for hop_key in sorted(config_percentages.keys(), key=lambda x: int(x.split('-')[0])):
        print(f"{hop_key}:")
        for template, pct in config_percentages[hop_key].items():
            print(f"  {template}: {pct:.1f}%")

    # 比较实际使用与配置
    compare_actual_vs_config(actual_percentages, config_percentages)

    # 打印汇总统计
    print_summary_statistics(actual_usage, actual_percentages)

    print("\n分析完成！")

if __name__ == "__main__":
    main()