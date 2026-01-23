#!/usr/bin/env python3
"""
验证路径脚本 - 检查forward_paths.jsonl中的路径是否有重复节点
"""

import json
import os
from typing import List, Dict, Set

def load_forward_paths(filepath: str) -> List[Dict]:
    """加载forward_paths.jsonl文件"""
    paths = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    path = json.loads(line.strip())
                    paths.append(path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"错误: 文件格式不正确 {filepath}: {e}")
        return []

    return paths

def check_path_uniqueness(path: Dict) -> Dict:
    """检查单个路径是否有重复节点"""
    path_id = path.get('path_id', 'unknown')
    path_length = path.get('path_length', 0)
    detailed_path = path.get('detailed_path', [])

    # 收集路径中的所有节点
    nodes_in_path = []
    edge_ids_in_path = []

    for edge_info in detailed_path:
        source_id = edge_info.get('source_id')
        target_id = edge_info.get('target_id')
        edge_id = edge_info.get('edge_id')

        nodes_in_path.extend([source_id, target_id])
        edge_ids_in_path.append(edge_id)

    # 检查节点重复
    node_count = len(nodes_in_path)
    unique_nodes = set(nodes_in_path)
    unique_node_count = len(unique_nodes)

    # 检查边重复
    unique_edges = set(edge_ids_in_path)
    unique_edge_count = len(unique_edges)

    # 路径应该有 path_length + 1 个唯一节点
    expected_unique_nodes = path_length + 1
    expected_unique_edges = path_length

    has_duplicate_nodes = unique_node_count != expected_unique_nodes
    has_duplicate_edges = unique_edge_count != expected_unique_edges

    # 找出重复的节点
    duplicate_nodes = []
    if has_duplicate_nodes:
        node_counts = {}
        for node in nodes_in_path:
            node_counts[node] = node_counts.get(node, 0) + 1

        duplicate_nodes = [(node, count) for node, count in node_counts.items() if count > 1]

    # 找出重复的边
    duplicate_edges = []
    if has_duplicate_edges:
        edge_counts = {}
        for edge in edge_ids_in_path:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        duplicate_edges = [(edge, count) for edge, count in edge_counts.items() if count > 1]

    return {
        'path_id': path_id,
        'path_length': path_length,
        'total_nodes': node_count,
        'unique_nodes': unique_node_count,
        'expected_unique_nodes': expected_unique_nodes,
        'total_edges': len(edge_ids_in_path),
        'unique_edges': unique_edge_count,
        'expected_unique_edges': expected_unique_edges,
        'has_duplicate_nodes': has_duplicate_nodes,
        'has_duplicate_edges': has_duplicate_edges,
        'duplicate_nodes': duplicate_nodes,
        'duplicate_edges': duplicate_edges,
        'is_valid': not has_duplicate_nodes and not has_duplicate_edges,
        'nodes_sequence': nodes_in_path,
        'edge_ids_sequence': edge_ids_in_path
    }

def analyze_path_patterns(paths: List[Dict]) -> Dict:
    """分析路径模式"""
    total_paths = len(paths)
    valid_paths = 0
    invalid_paths = 0

    path_length_stats = {}
    duplicate_node_stats = {}
    duplicate_edge_stats = {}

    invalid_path_details = []

    for path in paths:
        validation_result = check_path_uniqueness(path)

        if validation_result['is_valid']:
            valid_paths += 1
        else:
            invalid_paths += 1
            invalid_path_details.append(validation_result)

        # 统计路径长度分布
        path_length = validation_result['path_length']
        path_length_stats[path_length] = path_length_stats.get(path_length, 0) + 1

        # 统计重复节点情况
        if validation_result['has_duplicate_nodes']:
            dup_count = len(validation_result['duplicate_nodes'])
            duplicate_node_stats[dup_count] = duplicate_node_stats.get(dup_count, 0) + 1

        # 统计重复边情况
        if validation_result['has_duplicate_edges']:
            dup_count = len(validation_result['duplicate_edges'])
            duplicate_edge_stats[dup_count] = duplicate_edge_stats.get(dup_count, 0) + 1

    return {
        'total_paths': total_paths,
        'valid_paths': valid_paths,
        'invalid_paths': invalid_paths,
        'validity_rate': valid_paths / total_paths if total_paths > 0 else 0,
        'path_length_stats': path_length_stats,
        'duplicate_node_stats': duplicate_node_stats,
        'duplicate_edge_stats': duplicate_edge_stats,
        'invalid_path_details': invalid_path_details
    }

def print_detailed_report(analysis: Dict, show_invalid_details: bool = True, max_invalid_to_show: int = 10):
    """打印详细报告"""
    print("="*80)
    print("路径唯一性验证报告")
    print("="*80)

    print(f"总路径数: {analysis['total_paths']}")
    print(f"有效路径数: {analysis['valid_paths']}")
    print(f"无效路径数: {analysis['invalid_paths']}")
    print(f"有效率: {analysis['validity_rate']*100:.2f}%")

    print("\n路径长度分布:")
    for length, count in sorted(analysis['path_length_stats'].items()):
        print(f"  {length}-hop: {count} 条路径")

    if analysis['duplicate_node_stats']:
        print("\n重复节点统计:")
        for dup_count, path_count in sorted(analysis['duplicate_node_stats'].items()):
            print(f"  有 {dup_count} 个重复节点的路径: {path_count} 条")

    if analysis['duplicate_edge_stats']:
        print("\n重复边统计:")
        for dup_count, path_count in sorted(analysis['duplicate_edge_stats'].items()):
            print(f"  有 {dup_count} 条重复边的路径: {path_count} 条")

    if show_invalid_details and analysis['invalid_path_details']:
        print(f"\n无效路径详情 (显示前 {min(max_invalid_to_show, len(analysis['invalid_path_details']))} 条):")
        print("-"*80)

        for i, invalid_path in enumerate(analysis['invalid_path_details'][:max_invalid_to_show]):
            print(f"\n{i+1}. 路径ID: {invalid_path['path_id']}")
            print(f"   路径长度: {invalid_path['path_length']}")
            print(f"   节点: {invalid_path['total_nodes']} 个 (应有 {invalid_path['expected_unique_nodes']} 个唯一节点)")
            print(f"   边: {invalid_path['total_edges']} 条 (应有 {invalid_path['expected_unique_edges']} 条唯一边)")

            if invalid_path['duplicate_nodes']:
                print(f"   重复节点: {invalid_path['duplicate_nodes']}")

            if invalid_path['duplicate_edges']:
                print(f"   重复边: {invalid_path['duplicate_edges']}")

            print(f"   节点序列: {invalid_path['nodes_sequence']}")
            print(f"   边ID序列: {invalid_path['edge_ids_sequence']}")

def save_validation_report(analysis: Dict, output_file: str):
    """保存验证报告到文件"""
    report_data = {
        'summary': {
            'total_paths': analysis['total_paths'],
            'valid_paths': analysis['valid_paths'],
            'invalid_paths': analysis['invalid_paths'],
            'validity_rate': analysis['validity_rate']
        },
        'statistics': {
            'path_length_stats': analysis['path_length_stats'],
            'duplicate_node_stats': analysis['duplicate_node_stats'],
            'duplicate_edge_stats': analysis['duplicate_edge_stats']
        },
        'invalid_paths': analysis['invalid_path_details']
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存到: {output_file}")

def main():
    """主函数"""
    import sys

    # 默认文件路径
    default_input_file = "outputs/paths/forward_paths.jsonl"
    default_output_file = "outputs/summary/path_validation_report.json"

    # 支持命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input_file

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = default_output_file

    print(f"验证文件: {input_file}")

    # 加载路径数据
    paths = load_forward_paths(input_file)
    if not paths:
        print("未能加载路径数据，程序退出")
        return

    print(f"加载了 {len(paths)} 条路径")

    # 分析路径
    analysis = analyze_path_patterns(paths)

    # 打印报告
    print_detailed_report(analysis)

    # 保存详细报告
    save_validation_report(analysis, output_file)

    # 返回状态码
    if analysis['invalid_paths'] > 0:
        print(f"\n⚠️  警告: 发现 {analysis['invalid_paths']} 条无效路径（有重复节点或边）")
        sys.exit(1)
    else:
        print(f"\n✅ 所有路径都有效，没有发现重复节点或边")
        sys.exit(0)

if __name__ == "__main__":
    main()