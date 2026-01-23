#!/usr/bin/env python3
"""
改进版并行路径生成脚本，遵循CLAUDE.md中更新的要求
生成第三类问题路径，包括：
1. parallel numerical comparison paths
2. parallel node comparison paths
3. parallel different answer type paths
4. combo QA paths
"""
import json
import random
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple

def load_forward_qa_paths(filepath: str) -> List[Dict]:
    """从forward_qa.jsonl加载第一类问题路径"""
    paths = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            qa = json.loads(line.strip())
            # 确保包含必要的字段
            if 'path_edges' in qa and 'detailed_path' in qa and 'answer_type' in qa:
                paths.append(qa)
    return paths

def extract_nodes_and_edges(path: Dict) -> Tuple[Set[str], Set[str]]:
    """提取路径中的节点和边"""
    nodes = set()
    edges = set(path['path_edges'])

    for edge_info in path['detailed_path']:
        nodes.add(edge_info['source_id'])
        nodes.add(edge_info['target_id'])

    return nodes, edges

def are_paths_disjoint(path1_nodes: Set[str], path1_edges: Set[str],
                      path2_nodes: Set[str], path2_edges: Set[str]) -> bool:
    """检查两条路径是否不相交（无共同节点和边）"""
    return len(path1_nodes & path2_nodes) == 0 and len(path1_edges & path2_edges) == 0

def are_paths_globally_unique(path1_nodes: Set[str], path1_edges: Set[str],
                             path2_nodes: Set[str], path2_edges: Set[str],
                             global_used_path_sets: Set[frozenset]) -> bool:
    """检查两条路径的组合是否全局唯一（避免重复的path_edges组合）"""
    # 创建路径组合的规范表示
    path1_edges_frozen = frozenset(path1_edges)
    path2_edges_frozen = frozenset(path2_edges)

    # 创建两种可能的组合（因为path_edges是列表，顺序不重要）
    combination1 = frozenset([path1_edges_frozen, path2_edges_frozen])

    # 检查是否已存在相同的组合
    return combination1 not in global_used_path_sets

def find_parallel_paths_by_type(forward_paths: List[Dict],
                               same_type_count: int, diff_type_count: int,
                               path1_filter, path2_filter, path_type_name: str,
                               global_used_path_sets: Set[frozenset]) -> Tuple[List[Dict], List[Dict]]:
    """根据类型找到特定类型的并行路径，返回(相同类型, 不同类型)"""
    print(f"生成 {path_type_name} 路径...")

    # 按答案类型分组
    paths_by_type = defaultdict(list)
    path_info = {}

    for path in forward_paths:
        answer_type = path["answer_type"]
        paths_by_type[answer_type].append(path)

        nodes, edges = extract_nodes_and_edges(path)
        path_info[path['path_id']] = {
            'nodes': nodes,
            'edges': edges,
            'answer_type': answer_type,
            'path_length': path['path_length']
        }

    # 过滤出数量足够的答案类型，避免尝试组合数量太少的类型
    valid_answer_types = [at for at in paths_by_type.keys() if len(paths_by_type[at]) >= 5]
    print(f"  有效答案类型数量: {len(valid_answer_types)}/{len(paths_by_type.keys())}")

    # 更新paths_by_type只包含有效类型
    paths_by_type = {k: v for k, v in paths_by_type.items() if k in valid_answer_types}

    same_type_paths = []
    diff_type_paths = []
    used_path_ids = set()

    # 相同答案类型的路径组合
    print(f"  生成相同答案类型的 {path_type_name}...")
    for answer_type, paths in paths_by_type.items():
        type_paths = []

        # 筛选符合条件的路径
        filtered_paths = [p for p in paths if path1_filter(p)]
        for i in range(len(filtered_paths)):
            for j in range(i + 1, len(filtered_paths)):
                path1 = filtered_paths[i]
                path2 = filtered_paths[j]

                if path1['path_id'] in used_path_ids or path2['path_id'] in used_path_ids:
                    continue

                if not path2_filter(path2):
                    continue

                info1 = path_info[path1['path_id']]
                info2 = path_info[path2['path_id']]

                if are_paths_disjoint(info1['nodes'], info1['edges'], info2['nodes'], info2['edges']):
                    # 检查全局唯一性（避免重复的path_edges组合）
                    if are_paths_globally_unique(info1['nodes'], info1['edges'], info2['nodes'], info2['edges'], global_used_path_sets):
                        type_paths.append((path1, path2))
                        used_path_ids.add(path1['path_id'])
                        used_path_ids.add(path2['path_id'])

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(info1['edges'])
                        path2_edges_frozen = frozenset(info2['edges'])
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

                        if len(type_paths) >= same_type_count // len([t for t in paths_by_type.keys() if len(paths_by_type[t]) >= 2]):
                            break

        same_type_paths.extend(type_paths)

    # 重置used_path_ids来生成不同类型的路径
    used_path_ids = set()

    # 不同答案类型的路径组合
    print(f"  生成不同答案类型的 {path_type_name}...")
    answer_types = list(paths_by_type.keys())

    for i in range(len(answer_types)):
        for j in range(i + 1, len(answer_types)):
            type1 = answer_types[i]
            type2 = answer_types[j]

            paths1 = [p for p in paths_by_type[type1] if path1_filter(p) and p['path_id'] not in used_path_ids]
            paths2 = [p for p in paths_by_type[type2] if path2_filter(p) and p['path_id'] not in used_path_ids]

            if len(paths1) < 2 or len(paths2) < 2:
                continue

            max_attempts = min(len(paths1) * len(paths2), 1000)  # 增加尝试次数
            attempts = 0
            found_pairs = 0

            while attempts < max_attempts and found_pairs < diff_type_count:
                path1 = random.choice(paths1)
                path2 = random.choice(paths2)

                info1 = path_info[path1['path_id']]
                info2 = path_info[path2['path_id']]

                if are_paths_disjoint(info1['nodes'], info1['edges'], info2['nodes'], info2['edges']):
                    # 检查全局唯一性（避免重复的path_edges组合）
                    if are_paths_globally_unique(info1['nodes'], info1['edges'], info2['nodes'], info2['edges'], global_used_path_sets):
                        diff_type_paths.append((path1, path2))
                        used_path_ids.add(path1['path_id'])
                        used_path_ids.add(path2['path_id'])

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(info1['edges'])
                        path2_edges_frozen = frozenset(info2['edges'])
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

                        found_pairs += 1

                attempts += 1

            if found_pairs > 0:
                print(f"    {type1} + {type2}: 找到 {found_pairs} 对路径")

    return same_type_paths, diff_type_paths

def find_specific_hop_combinations(forward_paths: List[Dict], hop1: int, hop2: int,
                                 same_type_count: int, diff_type_count: int,
                                 global_used_path_sets: Set[frozenset]) -> Tuple[List[Dict], List[Dict]]:
    """找到特定跳数组合的并行路径，返回(相同类型, 不同类型)"""
    print(f"生成 {hop1}-hop + {hop2}-hop 路径...")

    # 按答案类型分组
    paths_by_type = defaultdict(list)
    path_info = {}

    for path in forward_paths:
        answer_type = path["answer_type"]
        paths_by_type[answer_type].append(path)

        nodes, edges = extract_nodes_and_edges(path)
        path_info[path['path_id']] = {
            'nodes': nodes,
            'edges': edges,
            'answer_type': answer_type,
            'path_length': path['path_length']
        }

    # 过滤出数量足够的答案类型
    valid_answer_types = [at for at in paths_by_type.keys() if len(paths_by_type[at]) >= 5]
    paths_by_type = {k: v for k, v in paths_by_type.items() if k in valid_answer_types}

    same_type_paths = []
    diff_type_paths = []
    used_path_ids = set()

    # 筛选指定跳数的路径
    hop1_paths = [p for p in forward_paths if p['path_length'] == hop1]
    hop2_paths = [p for p in forward_paths if p['path_length'] == hop2]

    # 相同答案类型的路径组合
    print(f"  生成相同答案类型的 {hop1}-hop + {hop2}-hop...")
    for answer_type, paths in paths_by_type.items():
        type_hop1_paths = [p for p in hop1_paths if p['answer_type'] == answer_type and p['path_id'] not in used_path_ids]
        type_hop2_paths = [p for p in hop2_paths if p['answer_type'] == answer_type and p['path_id'] not in used_path_ids]

        type_paths = []
        found_for_type = 0

        for path1 in type_hop1_paths:
            for path2 in type_hop2_paths:
                if path1['path_id'] in used_path_ids or path2['path_id'] in used_path_ids:
                    continue

                info1 = path_info[path1['path_id']]
                info2 = path_info[path2['path_id']]

                if are_paths_disjoint(info1['nodes'], info1['edges'], info2['nodes'], info2['edges']):
                    # 检查全局唯一性
                    if are_paths_globally_unique(info1['nodes'], info1['edges'], info2['nodes'], info2['edges'], global_used_path_sets):
                        type_paths.append((path1, path2))
                        used_path_ids.add(path1['path_id'])
                        used_path_ids.add(path2['path_id'])

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(info1['edges'])
                        path2_edges_frozen = frozenset(info2['edges'])
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

                        found_for_type += 1

                        if found_for_type >= same_type_count // len([t for t in paths_by_type.keys() if len([p for p in hop1_paths if p['answer_type'] == t]) >= 1 and len([p for p in hop2_paths if p['answer_type'] == t]) >= 1]):
                            break

        same_type_paths.extend(type_paths)

    # 重置used_path_ids来生成不同类型的路径
    used_path_ids = set()

    # 不同答案类型的路径组合
    print(f"  生成不同答案类型的 {hop1}-hop + {hop2}-hop...")
    answer_types = list(paths_by_type.keys())

    for i in range(len(answer_types)):
        for j in range(i + 1, len(answer_types)):
            type1 = answer_types[i]
            type2 = answer_types[j]

            paths1 = [p for p in hop1_paths if p['answer_type'] == type1 and p['path_id'] not in used_path_ids]
            paths2 = [p for p in hop2_paths if p['answer_type'] == type2 and p['path_id'] not in used_path_ids]

            max_attempts = min(len(paths1) * len(paths2), 200)
            attempts = 0
            found_pairs = 0

            while attempts < max_attempts and found_pairs < diff_type_count:
                path1 = random.choice(paths1)
                path2 = random.choice(paths2)

                info1 = path_info[path1['path_id']]
                info2 = path_info[path2['path_id']]

                if are_paths_disjoint(info1['nodes'], info1['edges'], info2['nodes'], info2['edges']):
                    # 检查全局唯一性（避免重复的path_edges组合）
                    if are_paths_globally_unique(info1['nodes'], info1['edges'], info2['nodes'], info2['edges'], global_used_path_sets):
                        diff_type_paths.append((path1, path2))
                        used_path_ids.add(path1['path_id'])
                        used_path_ids.add(path2['path_id'])

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(info1['edges'])
                        path2_edges_frozen = frozenset(info2['edges'])
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

                        found_pairs += 1

                attempts += 1

    return same_type_paths, diff_type_paths

def format_parallel_path(pair: Tuple[Dict, Dict], index: int, path_type: str, is_diff_type: bool = False, is_node_comparison: bool = False) -> Dict:
    """格式化并行路径"""
    path1, path2 = pair

    # 根据path1和path2的实际hop数量生成path_type字段
    hop1 = path1['path_length']
    hop2 = path2['path_length']

    if is_node_comparison:
        # 节点比较路径
        if hop1 == 1 and hop2 == 1:
            actual_path_type = "parallel node single-hop"
        elif hop1 == 1 and hop2 > 1:
            actual_path_type = f"parallel node single-hop + {hop2}-hop"
        elif hop1 > 1 and hop2 == 1:
            actual_path_type = f"parallel node single-hop + {hop1}-hop"
        else:
            actual_path_type = f"parallel node {hop1}-hop + {hop2}-hop"
    else:
        # 数值比较和不同类型路径
        if hop1 == 1 and hop2 == 1:
            actual_path_type = "parallel single-hop"
        elif hop1 == 1 and hop2 > 1:
            actual_path_type = f"parallel single-hop + {hop2}-hop"
        elif hop1 > 1 and hop2 == 1:
            actual_path_type = f"parallel single-hop + {hop1}-hop"
        else:
            actual_path_type = f"parallel {hop1}-hop + {hop2}-hop"

    # 生成path_id
    if is_node_comparison:
        path_id = f"parallel_same_node_{path_type}_{index + 1:04d}"
    else:
        path_id = f"parallel_{path_type}_{index + 1:04d}"

    # 对于不同类型的并行路径，answer_type是列表
    if is_diff_type:
        answer_type = [path1['answer_type'], path2['answer_type']]
    else:
        answer_type = path1['answer_type']  # 相同类型使用第一个答案的类型

    formatted_path = {
        'path_id': path_id,
        'path_type': actual_path_type,
        'original_path_ids': [path1['path_id'], path2['path_id']],
        'path_edges': [path1['path_edges'], path2['path_edges']],
        'path_description': [path1['path_description'], path2['path_description']],
        'answers': [path1['answer'], path2['answer']],
        'answer_type': answer_type
    }

    return formatted_path

def format_node_comparison_path(pair: Tuple[Dict, Dict], index: int, path_type: str, attr: str, attr_type: str, value1: str, value2: str) -> Dict:
    """格式化节点比较路径，包含共同数值属性信息"""
    path1, path2 = pair

    # 获取节点信息
    node1_id = path1['detailed_path'][-1]['target_id']
    node2_id = path2['detailed_path'][-1]['target_id']
    node_type = path1['answer_type']

    # 根据path1和path2的实际hop数量生成path_type字段
    hop1 = path1['path_length']
    hop2 = path2['path_length']

    if hop1 == 1 and hop2 == 1:
        actual_path_type = "parallel node single-hop"
    elif hop1 == 1 and hop2 > 1:
        actual_path_type = f"parallel node single-hop + {hop2}-hop"
    elif hop1 > 1 and hop2 == 1:
        actual_path_type = f"parallel node single-hop + {hop1}-hop"
    else:
        actual_path_type = f"parallel node {hop1}-hop + {hop2}-hop"

    path_id = f"parallel_same_node_{path_type}_{index + 1:04d}"

    formatted_path = {
        "path_id": path_id,
        "path_type": actual_path_type,
        "original_path_ids": [path1['path_id'], path2['path_id']],
        "path_edges": [path1['path_edges'], path2['path_edges']],
        "path_description": [path1['path_description'], path2['path_description']],
        "answer_entity_type": node_type,
        "answers": [path1['answer'], path2['answer']],
        "answer_nodes": [node1_id, node2_id],
        "same_numerical_attribute": [attr, attr],
        "same_numerical_attribute_type": [attr_type, attr_type],
        "same_numerical_attribute_value": [value1, value2]
    }

    return formatted_path

def find_common_numerical_attribute(node_type: str, path1: Dict, path2: Dict, nodes_data: Dict) -> List[Tuple]:
    """为两个相同类型的节点找到共同的数值属性，且不在路径中出现"""
    # 节点类型到数值属性的映射
    node_to_numerical_attrs = {
        "Person": ["birth_year"],
        "Company": ["number_of_employees", "number_of_departments", "number_of_products",
                   "franchise_count", "franchise_revenue", "founded_year"],
        "Country": ["population", "gdp", "total_area", "number_of_ethnic_groups"],
        "City": ["population", "gdp", "area", "number_of_universities", "number_of_companies"],
        "University": ["registered_students", "number_of_faculties", "number_of_professors",
                      "number_of_research_projects", "number_of_research_publications",
                      "number_of_research_awards", "number_of_research_patents",
                      "established_year", "university_rank"]
    }

    # 数值属性对应的type
    numerical_attr_types = {
        "birth_year": "Year",
        "number_of_employees": "Employee Count",
        "number_of_departments": "Department Count",
        "number_of_products": "Product Count",
        "franchise_count": "Franchise Count",
        "franchise_revenue": "Franchise Revenue",
        "founded_year": "Year",
        "population": "Population Count",
        "gdp": "GDP",
        "total_area": "Area",
        "number_of_ethnic_groups": "Ethnic Group Count",
        "area": "Area",
        "number_of_universities": "University Count",
        "number_of_companies": "Company Count",
        "registered_students": "Student Count",
        "number_of_faculties": "Faculty Count",
        "number_of_professors": "Professor Count",
        "number_of_research_projects": "Research Project Count",
        "number_of_research_publications": "Research Publication Count",
        "number_of_research_awards": "Research Award Count",
        "number_of_research_patents": "Research Patent Count",
        "established_year": "Year",
        "university_rank": "University Rank"
    }

    available_attrs = node_to_numerical_attrs.get(node_type, [])

    # 获取路径中已出现的属性类型
    path_types = set()
    for path in [path1, path2]:
        path_types.add(path['answer_type'])
        # 分析detailed_path来获取更多类型信息
        for edge_info in path['detailed_path']:
            if edge_info['source_type'] in numerical_attr_types.values():
                path_types.add(edge_info['source_type'])
            if edge_info['target_type'] in numerical_attr_types.values():
                path_types.add(edge_info['target_type'])

    # 找到不在路径中的共同数值属性
    common_attrs = []

    # 获取节点信息
    node1_id = path1['detailed_path'][-1]['target_id']
    node2_id = path2['detailed_path'][-1]['target_id']

    node1_data = nodes_data.get(node1_id, {}).get('attributes', {})
    node2_data = nodes_data.get(node2_id, {}).get('attributes', {})

    for attr in available_attrs:
        attr_type = numerical_attr_types[attr]
        if attr_type not in path_types:
            # 检查节点是否有这个属性
            if attr in node1_data and attr in node2_data:
                value1 = node1_data[attr].get('name', 'N/A')
                value2 = node2_data[attr].get('name', 'N/A')
                common_attrs.append((attr, attr_type, value1, value2))

    return common_attrs

def load_forward_qa_with_edges(filepath: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load forward_qa data and create edge-based mappings"""
    forward_qa_data = {}

    # Create mapping: target_name -> paths that end with this target_name
    target_to_paths = defaultdict(list)
    # Create mapping: source_name -> paths that start with this source_name
    source_to_paths = defaultdict(list)
    # Create edge lookup for quick access
    edge_data = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            qa = json.loads(line.strip())
            forward_qa_data[qa['path_id']] = qa

            detailed_path = qa.get('detailed_path', [])
            if detailed_path:
                # First edge source
                first_edge = detailed_path[0]
                source_name = first_edge.get('source_name')

                # Last edge target
                last_edge = detailed_path[-1]
                target_name = last_edge.get('target_name')

                if source_name:
                    source_to_paths[source_name].append(qa)
                if target_name:
                    target_to_paths[target_name].append(qa)

                # Store edge data for node checking
                for edge in detailed_path:
                    edge_data[edge['edge_id']] = edge

    return forward_qa_data, target_to_paths, source_to_paths, edge_data

def extract_node_ids_from_path(qa_data: Dict, edge_data: Dict) -> Set[str]:
    """Extract all node IDs from a path"""
    node_ids = set()
    detailed_path = qa_data.get('detailed_path', [])

    for edge in detailed_path:
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        if source_id:
            node_ids.add(source_id)
        if target_id:
            node_ids.add(target_id)

    return node_ids

def check_path_compatibility(path1: Dict, path2: Dict, edge_data: Dict) -> bool:
    """Check if two paths are compatible for combo (no shared nodes except connection point)"""
    nodes1 = extract_node_ids_from_path(path1, edge_data)
    nodes2 = extract_node_ids_from_path(path2, edge_data)

    # The intersection should be empty (no shared nodes except connection point)
    return len(nodes1.intersection(nodes2)) == 0

def generate_proper_combo_paths(forward_qa_data: Dict, target_to_paths: Dict, source_to_paths: Dict,
                               edge_data: Dict, target_counts: Dict[int, int]) -> List[Dict]:
    """Generate proper combo paths with correct connection logic"""
    combo_paths = []
    generated_combinations = set()

    print("Finding compatible path combinations...")

    # For each path as potential sub_path_1
    for path1_id, path1_data in forward_qa_data.items():
        detailed_path1 = path1_data.get('detailed_path', [])
        if not detailed_path1:
            continue

        # Get sub_path_1's last edge target_name
        last_edge = detailed_path1[-1]
        target_name = last_edge.get('target_name')

        if not target_name:
            continue

        # Find paths that start with this target_name
        for path2_data in source_to_paths.get(target_name, []):
            # Skip if it's the same path
            if path1_data['path_id'] == path2_data['path_id']:
                continue

            # Check if paths are compatible (no shared nodes except connection)
            if not check_path_compatibility(path1_data, path2_data, edge_data):
                continue

            total_length = path1_data['path_length'] + path2_data['path_length']

            # Check if we need this total length
            if total_length in target_counts and target_counts[total_length] > 0:
                # Create combination key to avoid duplicates
                combination_key = (path1_data['path_id'], path2_data['path_id'])
                if combination_key not in generated_combinations:
                    generated_combinations.add(combination_key)

                    # Verify the connection
                    detailed_path2 = path2_data.get('detailed_path', [])
                    if detailed_path2:
                        first_edge_2 = detailed_path2[0]
                        connection_match = first_edge_2.get('source_name') == target_name

                        if connection_match:
                            combo_path = {
                                'path_id': f"combo_combined_{total_length}_hop_{len(combo_paths) + 1}",
                                'path_type': f"combo {total_length}-hop",
                                'original_path_ids': [path1_data['path_id'], path2_data['path_id']],
                                'path_edges': [path1_data['path_edges'], path2_data['path_edges']],
                                'path_description': f"{path1_data.get('path_description', '')} + {path2_data.get('path_description', '')}",
                                'answers': [path1_data.get('answer', ''), path2_data.get('answer', '')],
                                'answer_types': [path1_data.get('answer_type', ''), path2_data.get('answer_type', '')],
                                'combination_method': 'path_combination',
                                'first_path_id': path1_data['path_id'],
                                'second_path_id': path2_data['path_id'],
                                'connection_point': target_name
                            }

                            combo_paths.append(combo_path)
                            target_counts[total_length] -= 1

                            
                            if target_counts[total_length] == 0:
                                del target_counts[total_length]
                                print(f"  Completed {total_length}-hop combinations")

    return combo_paths

def generate_combo_paths(combined_paths: List[Dict], hop_counts: List[int], combo_config: Dict) -> List[Dict]:
    """生成组合路径（只使用combo_combine方法）"""
    print("生成组合路径...")

    combo_paths = []

    # 使用正确的连接逻辑组合已有路径
    print("  使用combo_combine方法组合已有路径...")

    # 加载forward_paths数据以获取详细的路径信息
    forward_paths_file = '/Users/maxhe/Downloads/Research/Agent RL/mini-wiki/outputs/paths/forward_paths.jsonl'
    forward_qa_data, target_to_paths, source_to_paths, edge_data = load_forward_qa_with_edges(forward_paths_file)

    print(f"Loaded {len(forward_qa_data)} forward QA paths")
    print(f"Target mappings: {len(target_to_paths)} unique targets")
    print(f"Source mappings: {len(source_to_paths)} unique sources")

    # 目标计数用于不同hop组合 - 从config中获取
    target_counts = {
        7: combo_config.get('7_hop', 0),
        8: combo_config.get('8_hop', 0),
        9: combo_config.get('9_hop', 0),
        10: combo_config.get('10_hop', 0),
        11: combo_config.get('11_hop', 0),
        12: combo_config.get('12_hop', 0)
    }

    # 生成正确的combo路径
    proper_combo_paths = generate_proper_combo_paths(forward_qa_data, target_to_paths, source_to_paths, edge_data, target_counts)
    combo_paths.extend(proper_combo_paths)

    print(f"Generated {len(proper_combo_paths)} combo paths")

    return combo_paths

# 定义数值类型的answer_type
NUMERICAL_TYPES = {
    'Year', 'GDP', 'Population Count', 'Area', 'University Rank', 'Student Count',
    'Professor Count', 'Research Project Count', 'Research Publication Count', 'Research Award Count',
    'Research Patent Count', 'Employee Count', 'Department Count', 'Product Count',
    'Franchise Count', 'Franchise Revenue', 'University Count', 'Company Count', 'Ethic Group Count'
}

def is_numerical_type(answer_type):
    """检查answer_type是否为数值类型"""
    return answer_type in NUMERICAL_TYPES

def load_all_nodes():
    """加载所有节点数据"""
    nodes = {}
    node_files = [
        "outputs/nodes/nodes_persons.jsonl",
        "outputs/nodes/nodes_companies.jsonl",
        "outputs/nodes/nodes_countries.jsonl",
        "outputs/nodes/nodes_cities.jsonl",
        "outputs/nodes/nodes_universities.jsonl"
    ]

    for file_path in node_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    node = json.loads(line.strip())
                    nodes[node['node_id']] = node

    return nodes

def main():
    """主函数"""
    print("开始生成改进的第三类问题路径...")

    # 使用统一配置参数
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.unified_builder import get_complex_paths_config

    config = get_complex_paths_config()

    # 用于并行路径生成的多跳配置参数
    # 实际的多跳路径生成在generate_parallel_paths_by_types函数中实现

    # 加载数据 - 从forward_paths.jsonl加载
    forward_paths = load_forward_qa_paths('outputs/paths/forward_paths.jsonl')
    nodes_data = load_all_nodes()
    print(f"加载了 {len(forward_paths)} 条forward QA路径和 {len(nodes_data)} 个节点")

    # 过滤出长度不超过6的路径用于并行路径生成（combo路径使用所有路径）
    parallel_paths = [p for p in forward_paths if p['path_length'] <= 6]
    combo_paths_data = forward_paths  # combo路径可以使用所有长度的路径

    # 用于combo路径的组合：
    # - combo_combined方法需要sub-path不超过6-hop
    combo_sub_paths_for_combined = [p for p in forward_paths if p['path_length'] <= 6]

    # 定义节点类型
    node_types = ["Person", "Company", "Country", "City", "University"]

    # 初始化结果列表
    numerical_comparison_paths = []
    node_comparison_paths = []
    different_type_paths = []

    # 全局跟踪，用于避免重复的path_edges组合
    global_used_path_sets = set()

    # i. 生成 parallel numerical comparison paths (必须相同数值类型)
    print("=== 生成 parallel numerical comparison paths ===")

    # 按答案类型分组（只使用不超过6-hop的路径用于并行生成）
    paths_by_type = defaultdict(list)
    for path in parallel_paths:
        if is_numerical_type(path['answer_type']):
            paths_by_type[path['answer_type']].append(path)

    print(f"数值类型分布: {dict((k, len(v)) for k, v in paths_by_type.items())}")

    # 1. 两个都是single-hop，且相同数值类型
    # 降低要求：只要有至少1个single-hop路径就尝试
    for numerical_type in paths_by_type.keys():
        single_hop_type_paths = [p for p in paths_by_type[numerical_type] if p['path_length'] == 1]
        if len(single_hop_type_paths) >= 2:
            attempts = 0
            max_attempts = min(20, len(single_hop_type_paths) * 2)
            while attempts < max_attempts:
                path1, path2 = random.sample(single_hop_type_paths, 2)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(numerical_comparison_paths), "ss", False)
                        numerical_comparison_paths.append(formatted)

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)
                attempts += 1

    # 2. 一个single-hop，一个multi-hop，且相同数值类型
    for numerical_type in paths_by_type.keys():
        single_hop_type_paths = [p for p in paths_by_type[numerical_type] if p['path_length'] == 1]
        multi_hop_type_paths = [p for p in paths_by_type[numerical_type] if p['path_length'] > 1]

        if len(single_hop_type_paths) >= 1 and len(multi_hop_type_paths) >= 1:
            attempts = 0
            max_attempts = min(20, len(single_hop_type_paths) * len(multi_hop_type_paths))
            while attempts < max_attempts:
                path1 = random.choice(single_hop_type_paths)
                path2 = random.choice(multi_hop_type_paths)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(numerical_comparison_paths), "sm", False)
                        numerical_comparison_paths.append(formatted)

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)
                attempts += 1

    # 3. 两个multi-hop，且相同数值类型
    for numerical_type in paths_by_type.keys():
        multi_hop_type_paths = [p for p in paths_by_type[numerical_type] if p['path_length'] > 1]
        if len(multi_hop_type_paths) >= 2:
            attempts = 0
            max_attempts = min(20, len(multi_hop_type_paths) * 2)
            while attempts < max_attempts:
                path1, path2 = random.sample(multi_hop_type_paths, 2)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(numerical_comparison_paths), "mm", False)
                        numerical_comparison_paths.append(formatted)

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)
                attempts += 1

    print(f"生成了 {len(numerical_comparison_paths)} 条numerical comparison paths")

    # ii. 生成 parallel node comparison paths
    print("=== 生成 parallel node comparison paths ===")

    # 按节点类型分组
    node_paths_by_type = defaultdict(list)
    for path in forward_paths:
        if path['answer_type'] in node_types:
            node_paths_by_type[path['answer_type']].append(path)

    # 1. 两个都是single-hop
    for node_type in node_types:
        single_hop_node_paths = [p for p in node_paths_by_type[node_type] if p['path_length'] == 1]
        if len(single_hop_node_paths) >= 2:
            for i in range(min(config['parallel_same_type_entity']['single_hop'] // len(node_types), len(single_hop_node_paths) // 2)):
                path1, path2 = random.sample(single_hop_node_paths, 2)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        # 找到共同的数值属性
                        common_attrs = find_common_numerical_attribute(node_type, path1, path2, nodes_data)
                        if common_attrs:
                            selected_attr, attr_type, value1, value2 = random.choice(common_attrs)
                            formatted = format_node_comparison_path((path1, path2), len(node_comparison_paths), "ss", selected_attr, attr_type, value1, value2)
                            if formatted:
                                node_comparison_paths.append(formatted)

                                # 记录这个组合以确保唯一性
                                path1_edges_frozen = frozenset(edges1)
                                path2_edges_frozen = frozenset(edges2)
                                combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                                global_used_path_sets.add(combination)

    # 2. 一个single-hop，一个multi-hop
    for node_type in node_types:
        single_hop_node_paths = [p for p in node_paths_by_type[node_type] if p['path_length'] == 1]
        multi_hop_node_paths = [p for p in node_paths_by_type[node_type] if p['path_length'] > 1]

        for i in range(min(config['parallel_same_type_entity']['single_multi_hop'] // len(node_types), min(len(single_hop_node_paths), len(multi_hop_node_paths)))):
            path1 = random.choice(single_hop_node_paths)
            path2 = random.choice(multi_hop_node_paths)
            nodes1, edges1 = extract_nodes_and_edges(path1)
            nodes2, edges2 = extract_nodes_and_edges(path2)
            if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                # 检查全局唯一性
                if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                    # 找到共同的数值属性
                    common_attrs = find_common_numerical_attribute(node_type, path1, path2, nodes_data)
                    if common_attrs:
                        selected_attr, attr_type, value1, value2 = random.choice(common_attrs)
                        formatted = format_node_comparison_path((path1, path2), len(node_comparison_paths), "sm", selected_attr, attr_type, value1, value2)
                        if formatted:
                            node_comparison_paths.append(formatted)

                            # 记录这个组合以确保唯一性
                            path1_edges_frozen = frozenset(edges1)
                            path2_edges_frozen = frozenset(edges2)
                            combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                            global_used_path_sets.add(combination)

    # 3. 两个multi-hop
    for node_type in node_types:
        multi_hop_node_paths = [p for p in node_paths_by_type[node_type] if p['path_length'] > 1]
        if len(multi_hop_node_paths) >= 2:
            for i in range(min(config['parallel_same_type_entity']['multi_hop'] // len(node_types), len(multi_hop_node_paths) // 2)):
                path1, path2 = random.sample(multi_hop_node_paths, 2)
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        # 找到共同的数值属性
                        common_attrs = find_common_numerical_attribute(node_type, path1, path2, nodes_data)
                        if common_attrs:
                            selected_attr, attr_type, value1, value2 = random.choice(common_attrs)
                            formatted = format_node_comparison_path((path1, path2), len(node_comparison_paths), "mm", selected_attr, attr_type, value1, value2)
                            if formatted:
                                node_comparison_paths.append(formatted)

                                # 记录这个组合以确保唯一性
                                path1_edges_frozen = frozenset(edges1)
                                path2_edges_frozen = frozenset(edges2)
                                combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                                global_used_path_sets.add(combination)

    print(f"生成了 {len(node_comparison_paths)} 条node comparison paths")

    # iii. 生成 parallel different answer type paths
    print("=== 生成 parallel different answer type paths ===")

    # 1. 两个都是single-hop
    single_hop_all_paths = [p for p in forward_paths if p['path_length'] == 1]
    if len(single_hop_all_paths) >= 2:
        for i in range(min(config['parallel_different_type']['single_hop'], len(single_hop_all_paths) // 2)):
            path1, path2 = random.sample(single_hop_all_paths, 2)
            if path1['answer_type'] != path2['answer_type']:
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(different_type_paths), "ss", True)
                        different_type_paths.append(formatted)

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

    # 2. 一个single-hop，一个multi-hop
    single_hop_paths = [p for p in parallel_paths if p['path_length'] == 1]
    multi_hop_paths = [p for p in parallel_paths if p['path_length'] > 1]

    for i in range(min(config['parallel_different_type']['single_multi_hop'], min(len(single_hop_paths), len(multi_hop_paths)))):
        path1 = random.choice(single_hop_paths)
        path2 = random.choice(multi_hop_paths)
        if path1['answer_type'] != path2['answer_type']:
            nodes1, edges1 = extract_nodes_and_edges(path1)
            nodes2, edges2 = extract_nodes_and_edges(path2)
            if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                # 检查全局唯一性
                if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                    formatted = format_parallel_path((path1, path2), len(different_type_paths), "sm", True)
                    different_type_paths.append(formatted)

                    # 记录这个组合以确保唯一性
                    path1_edges_frozen = frozenset(edges1)
                    path2_edges_frozen = frozenset(edges2)
                    combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                    global_used_path_sets.add(combination)

    # 3. 两个multi-hop
    if len(multi_hop_paths) >= 2:
        for i in range(min(config['parallel_different_type']['multi_hop'], len(multi_hop_paths) // 2)):
            path1, path2 = random.sample(multi_hop_paths, 2)
            if path1['answer_type'] != path2['answer_type']:
                nodes1, edges1 = extract_nodes_and_edges(path1)
                nodes2, edges2 = extract_nodes_and_edges(path2)
                if are_paths_disjoint(nodes1, edges1, nodes2, edges2):
                    # 检查全局唯一性
                    if are_paths_globally_unique(nodes1, edges1, nodes2, edges2, global_used_path_sets):
                        formatted = format_parallel_path((path1, path2), len(different_type_paths), "mm", True)
                        different_type_paths.append(formatted)

                        # 记录这个组合以确保唯一性
                        path1_edges_frozen = frozenset(edges1)
                        path2_edges_frozen = frozenset(edges2)
                        combination = frozenset([path1_edges_frozen, path2_edges_frozen])
                        global_used_path_sets.add(combination)

    print(f"生成了 {len(different_type_paths)} 条different type paths")

    # iv. 生成 combo paths
    print("=== 生成 combo paths ===")
    combo_paths = generate_combo_paths(combo_sub_paths_for_combined, [7, 8, 9, 10, 11, 12], config['combo']) 

    # 按照hop数量从大到小排序combo路径
    def get_sort_key(combo_path):
        # 从path_type中提取hop数量，如"combo 12-hop" -> 12
        import re
        match = re.search(r'combo (\d+)-hop', combo_path['path_type'])
        hop_count = int(match.group(1)) if match else 0
        return hop_count

    combo_paths_sorted = sorted(combo_paths, key=get_sort_key, reverse=True)
    print(f"生成了 {len(combo_paths_sorted)} 条combo paths（已按hop数量排序）")

    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"Numerical comparison paths: {len(numerical_comparison_paths)}")
    print(f"Node comparison paths: {len(node_comparison_paths)}")
    print(f"Different type paths: {len(different_type_paths)}")
    print(f"Combo paths: {len(combo_paths)}")
    print(f"总计: {len(numerical_comparison_paths) + len(node_comparison_paths) + len(different_type_paths) + len(combo_paths)}")

    # 保存路径
    os.makedirs('outputs/complex_paths', exist_ok=True)

    # 保存并行数值比较路径
    with open('outputs/complex_paths/parallel_same_type_numerical_qa_paths.jsonl', 'w', encoding='utf-8') as f:
        for path in numerical_comparison_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # 保存并行节点比较路径
    with open('outputs/complex_paths/parallel_same_type_entity_qa_paths.jsonl', 'w', encoding='utf-8') as f:
        for path in node_comparison_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # 保存并行不同类型路径
    with open('outputs/complex_paths/parallel_diff_type_qa_paths.jsonl', 'w', encoding='utf-8') as f:
        for path in different_type_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # 保存combo路径（使用排序后的路径）
    with open('outputs/complex_paths/combo_qa_paths.jsonl', 'w', encoding='utf-8') as f:
        for path in combo_paths_sorted:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # 保存汇总
    all_paths = numerical_comparison_paths + node_comparison_paths + different_type_paths + combo_paths_sorted
    summary = {
        "parallel_same_type_numerical": len(numerical_comparison_paths),
        "parallel_same_type_node": len(node_comparison_paths),
        "parallel_diff_type": len(different_type_paths),
        "combo_qa": len(combo_paths_sorted),
        "total": len(all_paths),
        "numerical_types": list(NUMERICAL_TYPES),
        "node_types": node_types
    }

    with open('outputs/complex_paths/classified_paths_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("改进的第三类问题路径生成完成!")

if __name__ == "__main__":
    main()