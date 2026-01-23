#!/usr/bin/env python3
"""
高效版本的知识图谱构建脚本，减少内存使用
"""
import sys
import os
import json
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_utils import load_schema

def load_nodes_batch(filepath, batch_size=50):
    """分批加载节点以减少内存使用"""
    nodes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                node = json.loads(line.strip())
                nodes.append(node)
                # 当达到批次大小时，返回这批节点
                if len(nodes) >= batch_size:
                    yield nodes
                    nodes = []
            # 返回剩余的节点
            if nodes:
                yield nodes
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确 {filepath}")
        return

def load_node_attributes():
    """加载所有节点的attributes_in_abstracts映射"""
    print("加载节点attributes_in_abstracts映射...")

    node_attributes = {}
    file_paths = [
        'outputs/nodes/nodes_countries.jsonl',
        'outputs/nodes/nodes_cities.jsonl',
        'outputs/nodes/nodes_universities.jsonl',
        'outputs/nodes/nodes_companies.jsonl',
        'outputs/nodes/nodes_persons.jsonl'
    ]

    for filepath in file_paths:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        node = json.loads(line.strip())
                        node_id = node['node_id']

                        # 获取attributes_in_abstracts中的属性名
                        abstracts_attrs = set()
                        if 'attributes_in_abstracts' in node:
                            for attr_name in node['attributes_in_abstracts'].keys():
                                abstracts_attrs.add(attr_name)

                        node_attributes[node_id] = abstracts_attrs
            except Exception as e:
                print(f"警告: 加载 {filepath} 时出错: {e}")

    print(f"加载了 {len(node_attributes)} 个节点的attributes信息")
    return node_attributes

def check_need_access_for_edge(edge, node_attributes):
    """检查edge的need_access状态（与access相反）"""
    source_id = edge['source_id']
    target_id = edge['target_id']
    edge_type = edge['edge_type']
    source_type = edge['source_type']

    # 如果source是实体节点 (City, Country, Person, Company, University)
    if source_type in ['City', 'Country', 'Person', 'Company', 'University']:
        # 检查edge_type是否在source节点的attributes_in_abstracts中
        if source_id in node_attributes:
            # 如果在attributes_in_abstracts中，则need_access=False，否则为True
            return edge_type not in node_attributes[source_id]

    # 如果target是实体节点，需要查找原始edge_type
    if target_id in node_attributes:
        # 获取原始edge_type（去掉_of后缀）
        original_edge_type = edge_type
        if edge_type.endswith('_of'):
            original_edge_type = edge_type[:-3]

        # 特殊处理一些反向关系
        reverse_to_original = {
            'birth_year_of': 'birth_year',
            'birth_city_of': 'birth_city',
            'current_living_city_of': 'current_living_city',
            'university_attended_by': 'university_attended',
            'current_working_company_of': 'current_working_company',
            'father_of': 'father',
            'mother_of': 'mother',
            'spouse_of': 'spouse',
            'leader_of_country_for': 'leader_of_country',
            'mayor_of_city_for': 'mayor_of_city',
            'best_friend_of': 'best_friend',
            'sister_country_of': 'sister_country',
            'sister_city_of': 'sister_city',
            'located_country_of': 'located_country',
            'capital_city': 'capital_of',
            'located_city_of': 'located_city',
            'headquarter_city_of': 'headquarter_city'
        }

        if edge_type in reverse_to_original:
            original_edge_type = reverse_to_original[edge_type]

        # 如果在attributes_in_abstracts中，则need_access=False，否则为True
        return original_edge_type not in node_attributes[target_id]

    # 如果无法检查，默认为True（需要访问）
    return True

def get_reverse_relation_type(relation_type):
    """获取反向关系类型"""
    reverse_mapping = {
        'located_country': 'located_country_of',
        'capital_of': 'capital_city',
        'located_city': 'located_city_of',
        'headquarter_city': 'headquarter_city_of',
        'birth_city': 'birth_city_of',
        'current_living_city': 'current_living_city_of',
        'university_attended': 'university_attended_by',
        'current_working_company': 'current_working_company_of',
        'father': 'father_of',
        'mother': 'mother_of',
        'spouse': 'spouse_of',
        'leader_of_country': 'leader_of_country_for',
        'mayor_of_city': 'mayor_of_city_for',
        'best_friend': 'best_friend_of',
        'sister_country': 'sister_country_of',
        'sister_city': 'sister_city_of'
    }

    if relation_type in reverse_mapping:
        return reverse_mapping[relation_type]
    elif not relation_type.endswith('_of'):
        return relation_type + '_of'
    else:
        return relation_type[:-3]

def get_reverse_mapping_type(mapping_type):
    """反转映射类型"""
    if mapping_type == '1-1':
        return '1-1'
    elif mapping_type == '1-n':
        return 'n-1'
    elif mapping_type == 'n-1':
        return '1-n'
    else:
        return mapping_type

def build_forward_edges():
    """构建正向关系"""
    print("开始构建正向关系...")

    # 加载schema配置
    schema = load_schema('schema_config.json')

    # 构建属性到映射类型的映射
    attr_mapping = {}
    for entity_type in ['Country', 'City', 'University', 'Company', 'Person']:
        attrs = schema.get(f'{entity_type}_Attributes', [])
        for attr in attrs:
            attr_name = attr['name']
            mapping_type = attr['type'][2] if len(attr['type']) > 2 else '1-1'
            attr_mapping[attr_name] = mapping_type

    # 加载节点attributes_in_abstracts映射
    node_attributes = load_node_attributes()

    forward_edges = []
    edge_id_counter = 0

    # 文件路径
    file_paths = [
        'outputs/nodes/nodes_countries.jsonl',
        'outputs/nodes/nodes_cities.jsonl',
        'outputs/nodes/nodes_universities.jsonl',
        'outputs/nodes/nodes_companies.jsonl',
        'outputs/nodes/nodes_persons.jsonl'
    ]

    # 创建实体映射（只存储必要的ID和name）
    entity_map = {}

    # 第一遍：构建实体映射
    print("构建实体映射...")
    for filepath in file_paths:
        if os.path.exists(filepath):
            for batch in load_nodes_batch(filepath, 100):
                for node in batch:
                    entity_map[node['node_id']] = {
                        'name': node['name'],
                        'type': node['node_type']
                    }

    print(f"加载了 {len(entity_map)} 个实体")
    print(f"属性映射配置: {attr_mapping}")

    # 第二遍：构建边
    print("构建边...")
    for filepath in file_paths:
        if os.path.exists(filepath):
            for batch_idx, batch in enumerate(load_nodes_batch(filepath, 50)):
                print(f"  处理 {filepath} 批次 {batch_idx + 1}...")

                for node in batch:
                    node_type = node['node_type']
                    attributes = node['attributes']

                    for attr_name, attr_value in attributes.items():
                        # 处理实体属性
                        if isinstance(attr_value, dict) and 'node_id' in attr_value:
                            target_node_id = attr_value['node_id']
                            if target_node_id in entity_map:
                                target_entity = entity_map[target_node_id]

                                # 获取正确的映射类型
                                mapping_type = attr_mapping.get(attr_name, '1-1')

                                edge = {
                                    'edge_id': f'e_{edge_id_counter}',
                                    'edge_type': attr_name,
                                    'source_id': node['node_id'],
                                    'source_type': node_type,
                                    'source_name': node['name'],
                                    'target_id': target_node_id,
                                    'target_type': target_entity['type'],
                                    'target_name': target_entity['name'],
                                    'mapping_type': mapping_type
                                }

                                # 添加need_access字段
                                edge['need_access'] = check_need_access_for_edge(edge, node_attributes)

                                forward_edges.append(edge)
                                edge_id_counter += 1

                        # 处理非实体属性
                        elif isinstance(attr_value, dict) and 'name' in attr_value:
                            attr_value_name = attr_value['name']
                            target_type = attr_value.get('type', 'Value')

                            # 获取正确的映射类型
                            mapping_type = attr_mapping.get(attr_name, '1-1')

                            edge = {
                                'edge_id': f'e_{edge_id_counter}',
                                'edge_type': attr_name,
                                'source_id': node['node_id'],
                                'source_type': node_type,
                                'source_name': node['name'],
                                'target_id': f"attr_{attr_name}_{attr_value_name}",
                                'target_type': target_type,
                                'target_name': str(attr_value_name),
                                'mapping_type': mapping_type
                            }

                            # 添加need_access字段
                            edge['need_access'] = check_need_access_for_edge(edge, node_attributes)

                            forward_edges.append(edge)
                            edge_id_counter += 1

    print(f"正向关系构建完成，共 {len(forward_edges)} 条关系")
    return forward_edges

def build_reverse_edges(forward_edges, node_attributes):
    """构建反向关系"""
    print("开始构建反向关系...")

    reverse_edges = []
    edge_id_counter = len(forward_edges)

    # 构建映射统计
    print("构建映射统计...")
    reverse_mapping_stats = defaultdict(set)

    for edge in forward_edges:
        reverse_edge_type = get_reverse_relation_type(edge['edge_type'])
        reverse_mapping_stats[(edge['target_id'], reverse_edge_type)].add(edge['source_id'])

    print("生成反向关系...")
    for i, edge in enumerate(forward_edges):
        if i % 500 == 0:
            print(f"  处理反向关系 {i+1}/{len(forward_edges)}...")

        reverse_edge_type = get_reverse_relation_type(edge['edge_type'])

        # 检查实际的映射关系
        key = (edge['target_id'], reverse_edge_type)
        actual_targets = reverse_mapping_stats.get(key, set())

        # 根据实际情况确定映射类型
        if len(actual_targets) == 1:
            actual_reverse_mapping = '1-1'
        else:
            actual_reverse_mapping = '1-n'

        reverse_edge = {
            'edge_id': f'e_{edge_id_counter}',
            'edge_type': reverse_edge_type,
            'source_id': edge['target_id'],
            'source_type': edge['target_type'],
            'source_name': edge['target_name'],
            'target_id': edge['source_id'],
            'target_type': edge['source_type'],
            'target_name': edge['source_name'],
            'mapping_type': actual_reverse_mapping
        }

        # 添加need_access字段
        reverse_edge['need_access'] = check_need_access_for_edge(reverse_edge, node_attributes)

        reverse_edges.append(reverse_edge)
        edge_id_counter += 1

    print(f"反向关系构建完成，共 {len(reverse_edges)} 条关系")
    return reverse_edges

def save_edges(edges, filename):
    """保存边到文件"""
    print(f"保存边到 {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')

def generate_summary(forward_edges, reverse_edges):
    """生成摘要"""
    all_edges = forward_edges + reverse_edges

    # 统计信息
    mapping_stats = defaultdict(int)
    edge_types = defaultdict(int)
    source_types = defaultdict(int)
    need_access_stats = defaultdict(int)

    for edge in all_edges:
        mapping_stats[edge['mapping_type']] += 1
        edge_types[edge['edge_type']] += 1
        source_types[edge['source_type']] += 1

        # 统计need_access状态
        need_access_status = edge.get('need_access', True)
        need_access_stats['need_access_true' if need_access_status else 'need_access_false'] += 1

    summary = {
        'total_edges': len(all_edges),
        'forward_edges': len(forward_edges),
        'reverse_edges': len(reverse_edges),
        'mapping_types': dict(mapping_stats),
        'edge_types': dict(edge_types),
        'source_types': dict(source_types),
        'need_access_stats': dict(need_access_stats)
    }

    return summary

def main():
    """主函数"""
    print("开始构建高效知识图谱...")

    # 构建正向关系
    forward_edges = build_forward_edges()

    # 保存正向关系
    save_edges(forward_edges, 'outputs/edges/forward_edges.jsonl')

    # 加载节点attributes映射（需要传递给反向边构建函数）
    node_attributes = load_node_attributes()

    # 构建反向关系
    reverse_edges = build_reverse_edges(forward_edges, node_attributes)

    # 保存反向关系
    save_edges(reverse_edges, 'outputs/edges/reverse_edges.jsonl')

    # 保存所有边
    all_edges = forward_edges + reverse_edges
    save_edges(all_edges, 'outputs/edges/all_edges.jsonl')

    # 生成并保存摘要
    summary = generate_summary(forward_edges, reverse_edges)
    save_edges([summary], 'outputs/edges/graph_summary.json')

    print(f"知识图谱构建完成!")
    print(f"  总关系数: {summary['total_edges']}")
    print(f"  正向关系: {summary['forward_edges']}")
    print(f"  反向关系: {summary['reverse_edges']}")
    print(f"  映射类型: {summary['mapping_types']}")

    # 显示need_access统计
    need_access_stats = summary.get('need_access_stats', {})
    need_access_true = need_access_stats.get('need_access_true', 0)
    need_access_false = need_access_stats.get('need_access_false', 0)
    print(f"  Need_Access状态:")
    print(f"    Need_Access=True:  {need_access_true} ({need_access_true/summary['total_edges']*100:.1f}%)")
    print(f"    Need_Access=False: {need_access_false} ({need_access_false/summary['total_edges']*100:.1f}%)")

if __name__ == "__main__":
    main()