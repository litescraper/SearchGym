#!/usr/bin/env python3
"""
路径搜索策略标注脚本
为每条路径的每个步骤判断abstract是否包含足够信息，决定是否需要点击访问
"""
import json
import os
from typing import Dict, List, Any
from collections import defaultdict

def load_abstracts_info(abstracts_filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    加载abstract信息

    Args:
        abstracts_filepath: wiki_abstracts.json文件路径

    Returns:
        Dict: 实体ID到abstract信息的映射
    """
    try:
        with open(abstracts_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {abstracts_filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {abstracts_filepath}")
        return {}

def load_edges_info(edges_filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    加载边信息

    Args:
        edges_filepath: all_edges.jsonl文件路径

    Returns:
        Dict: 边ID到边信息的映射
    """
    edges = {}
    try:
        with open(edges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                edge = json.loads(line.strip())
                edges[edge['edge_id']] = edge
    except FileNotFoundError:
        print(f"错误: 文件未找到 {edges_filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {edges_filepath}")
        return {}

    return edges

def relation_keywords_in_abstract(relation_type: str) -> List[str]:
    """
    获取关系类型对应的关键词列表，用于检查abstract是否包含相关信息

    Args:
        relation_type: 关系类型

    Returns:
        List[str]: 关键词列表
    """
    # 关系类型到关键词的映射
    relation_mapping = {
        'birth_year': ['born', 'birth', 'year', 'age'],
        'birth_city': ['born', 'birth', 'city'],
        'current_living_city': ['living', 'resides', 'city'],
        'university_attended': ['university', 'college', 'education', 'attended'],
        'current_working_company': ['work', 'company', 'employed', 'job'],
        'father': ['father', 'dad'],
        'mother': ['mother', 'mom'],
        'spouse': ['spouse', 'husband', 'wife', 'married'],
        'leader_of_country': ['leader', 'president', 'prime minister', 'ruler'],
        'mayor_of_city': ['mayor'],
        'best_friend': ['friend', 'friendship'],
        'located_country': ['country', 'located', 'nation'],
        'capital_of': ['capital'],
        'located_city': ['city', 'located'],
        'headquarter_city': ['headquarter', 'city'],
        'population': ['population', 'people'],
        'gdp': ['gdp', 'economy'],
        'established_year': ['established', 'founded', 'year'],
        'founded_year': ['founded', 'established', 'year'],
        'sister_country': ['sister', 'country'],
        'sister_city': ['sister', 'city'],
    }

    return relation_mapping.get(relation_type, [relation_type.replace('_', ' ')])

def check_abstract_contains_relation(entity_id: str, relation_type: str, abstracts_info: Dict) -> bool:
    """
    检查实体的abstract是否包含特定关系的信息

    Args:
        entity_id: 实体ID
        relation_type: 关系类型
        abstracts_info: abstract信息字典

    Returns:
        bool: abstract是否包含足够的关系信息
    """
    if entity_id not in abstracts_info:
        return False

    entity_info = abstracts_info[entity_id]
    abstract_text = entity_info['abstract'].lower()
    abstract_attributes = entity_info.get('abstract_attributes', [])

    # 方法1：检查abstract_attributes是否直接包含该关系
    relation_base = relation_type
    if relation_base in abstract_attributes:
        return True

    # 方法2：检查abstract文本中是否包含关键词
    keywords = relation_keywords_in_abstract(relation_type)
    for keyword in keywords:
        if keyword in abstract_text:
            return True

    return False

def annotate_path_search_strategy(path_data: Dict, abstracts_info: Dict, edges_info: Dict) -> Dict:
    """
    为单条路径标注搜索策略

    Args:
        path_data: 路径数据
        abstracts_info: abstract信息
        edges_info: 边信息

    Returns:
        Dict: 标注后的路径数据
    """
    path_edges = path_data.get('path_edges', [])
    search_strategy = []

    # 获取路径的详细信息
    detailed_path = path_data.get('detailed_path', [])

    for i, edge_id in enumerate(path_edges):
        if edge_id not in edges_info:
            continue

        edge = edges_info[edge_id]
        source_id = edge['source_id']
        relation_type = edge['edge_type']
        target_name = edge['target_name']

        # 检查abstract是否包含足够信息
        sufficient_in_abstract = check_abstract_contains_relation(source_id, relation_type, abstracts_info)

        strategy_step = {
            "step": i,
            "entity": source_id,
            "relation": relation_type,
            "target": target_name,
            "sufficient_in_abstract": sufficient_in_abstract,
            "need_click": not sufficient_in_abstract
        }

        search_strategy.append(strategy_step)

    # 复制原始路径数据并添加搜索策略
    annotated_path = path_data.copy()
    annotated_path['search_strategy'] = search_strategy

    # 添加总体统计信息
    total_steps = len(search_strategy)
    need_click_steps = sum(1 for step in search_strategy if step['need_click'])
    annotated_path['search_stats'] = {
        'total_steps': total_steps,
        'need_click_steps': need_click_steps,
        'abstract_sufficient_steps': total_steps - need_click_steps
    }

    return annotated_path

def annotate_paths_file(input_filepath: str, output_filepath: str, abstracts_info: Dict, edges_info: Dict):
    """
    标注整个路径文件

    Args:
        input_filepath: 输入路径文件
        output_filepath: 输出路径文件
        abstracts_info: abstract信息
        edges_info: 边信息
    """
    annotated_paths = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                path_data = json.loads(line.strip())
                annotated_path = annotate_path_search_strategy(path_data, abstracts_info, edges_info)
                annotated_paths.append(annotated_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {input_filepath}")
        return

    # 保存标注结果
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for path in annotated_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    print(f"标注完成，结果保存到 {output_filepath}")

    # 统计信息
    total_paths = len(annotated_paths)
    paths_need_click = sum(1 for path in annotated_paths if path['search_stats']['need_click_steps'] > 0)
    avg_clicks = sum(path['search_stats']['need_click_steps'] for path in annotated_paths) / total_paths if total_paths > 0 else 0

    print(f"统计信息:")
    print(f"  总路径数: {total_paths}")
    print(f"  需要点击的路径数: {paths_need_click}")
    print(f"  平均点击次数: {avg_clicks:.2f}")

def main():
    """主函数"""
    # 文件路径
    abstracts_filepath = 'outputs/wiki/wiki_abstracts.json'
    edges_filepath = 'outputs/edges/all_edges.jsonl'

    # 需要标注的路径文件
    path_files = [
        ('outputs/paths/forward_paths.jsonl', 'outputs/qa_paths_with_search_strategy/forward_paths_annotated.jsonl'),
        ('outputs/paths/reverse_paths.jsonl', 'outputs/qa_paths_with_search_strategy/reverse_paths_annotated.jsonl'),
        ('outputs/complex_paths/parallel_same_type_qa_paths.jsonl', 'outputs/qa_paths_with_search_strategy/parallel_same_type_annotated.jsonl'),
        ('outputs/complex_paths/parallel_diff_type_qa_paths.jsonl', 'outputs/qa_paths_with_search_strategy/parallel_diff_type_annotated.jsonl'),
        ('outputs/complex_paths/granular_combo_qa_paths.jsonl', 'outputs/qa_paths_with_search_strategy/combo_paths_annotated.jsonl'),
    ]

    # 加载数据
    print("加载数据...")
    abstracts_info = load_abstracts_info(abstracts_filepath)
    edges_info = load_edges_info(edges_filepath)

    if not abstracts_info:
        print("错误: 未能加载abstract信息")
        return

    if not edges_info:
        print("错误: 未能加载边信息")
        return

    print(f"加载了 {len(abstracts_info)} 个实体的abstract信息")
    print(f"加载了 {len(edges_info)} 条边信息")

    # 标注所有路径文件
    for input_file, output_file in path_files:
        if os.path.exists(input_file):
            print(f"\n标注文件: {input_file}")
            annotate_paths_file(input_file, output_file, abstracts_info, edges_info)
        else:
            print(f"跳过不存在的文件: {input_file}")

    print("\n所有路径标注完成!")

if __name__ == "__main__":
    main()