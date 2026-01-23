#!/usr/bin/env python3
"""
改进版路径生成脚本，解决所有提到的问题
"""
import sys
import os
import json
import random
from collections import defaultdict, deque

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_edges(filepath):
    """加载边数据"""
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            edge = json.loads(line.strip())
            edges.append(edge)
    return edges

def load_nodes(filepath):
    """加载节点数据"""
    nodes = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            node = json.loads(line.strip())
            nodes[node['node_id']] = node
    return nodes

def build_graph(edges):
    """构建图结构"""
    graph = defaultdict(list)
    edge_info = {}
    target_types = {}

    for edge in edges:
        source_id = edge['source_id']
        target_id = edge['target_id']
        edge_type = edge['edge_type']
        mapping_type = edge['mapping_type']
        target_types[target_id] = edge['target_type']

        graph[source_id].append((target_id, edge_type, mapping_type))
        edge_info[(source_id, target_id, edge_type)] = edge

    return graph, edge_info, target_types

def get_node_name(node_id, nodes):
    """获取节点名称"""
    if node_id in nodes:
        return nodes[node_id]['name']
    elif node_id.startswith('attr_'):
        # 对于属性节点，提取属性值作为名称
        parts = node_id.split('_')
        return parts[-1] if len(parts) > 1 else node_id
    else:
        return node_id

def find_forward_paths(graph, edge_info, nodes, samples_per_hop_config):
    """查找第一类问题路径（不包含1-n关系）"""
    print("查找第一类问题路径...")

    # 只使用1-1和n-1的边
    valid_edges = [edge for edge in edge_info.values() if edge['mapping_type'] in ['1-1', 'n-1']]
    print(f"有效边数量 (1-1 + n-1): {len(valid_edges)}")

    forward_paths = []
    path_signatures = set()  # 用于确保路径唯一性

    # 寻找最大跳数
    min_hop = min(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())
    max_hop = max(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())

    # 为每个跳数生成路径)
    for hop_length in range(min_hop, max_hop + 1):
        print(f"生成 {hop_length}-hop 路径...")
        hop_paths = []

        # 确定当前hop的目标数量
        if samples_per_hop_config and f"{hop_length}-hop" in samples_per_hop_config:
            target_count = samples_per_hop_config[f"{hop_length}-hop"]
        else:
            continue

        # 对所有hop使用相同的DFS方法
        start_nodes = list(graph.keys())
        random.shuffle(start_nodes)

        # 增加尝试的节点数以提高成功率
        max_attempts = len(start_nodes)*3
        attempts = 0

        for start_node in start_nodes:
            if attempts >= max_attempts or len(hop_paths) >= target_count:
                break

            path = find_single_forward_path(
                graph, start_node, hop_length, edge_info, nodes, path_signatures
            )

            if path:
                hop_paths.append(path)

            attempts += 1

        forward_paths.extend(hop_paths)
        print(f"  {hop_length}-hop 路径: 生成 {len(hop_paths)} 条 (目标: {target_count})")

    return forward_paths

def find_forward_paths_by_access(graph, edge_info, nodes, samples_per_hop_config, need_access=True):
    """查找第一类问题路径，按need_access分类"""
    access_type = "need_access" if need_access else "no_need_access"
    print(f"查找第一类问题路径 ({access_type})...")

    # 根据need_access要求过滤有效边
    if need_access:
        # 至少有一个edge的need_access为true
        valid_edges = [edge for edge in edge_info.values()
                      if edge['mapping_type'] in ['1-1', 'n-1'] and edge.get('need_access', False)]
    else:
        # 所有edge的need_access都为false
        valid_edges = [edge for edge in edge_info.values()
                      if edge['mapping_type'] in ['1-1', 'n-1'] and not edge.get('need_access', False)]

    print(f"有效边数量 (1-1 + n-1, {access_type}): {len(valid_edges)}")

    forward_paths = []
    path_signatures = set()

    # 寻找最大跳数
    min_hop = min(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())
    max_hop = max(int(hop.split('-')[0]) for hop in samples_per_hop_config.keys())

    # 为每个跳数生成路径
    for hop_length in range(min_hop, max_hop + 1):
        print(f"生成 {hop_length}-hop 路径 ({access_type})...")
        hop_paths = []

        # 确定当前hop的目标数量
        if samples_per_hop_config and f"{hop_length}-hop" in samples_per_hop_config:
            target_count = samples_per_hop_config[f"{hop_length}-hop"]
        else:
            continue

        # 对所有hop使用相同的DFS方法
        start_nodes = list(graph.keys())
        random.shuffle(start_nodes)

        # 增加尝试的节点数以提高成功率
        max_attempts = len(start_nodes)*3
        attempts = 0

        for start_node in start_nodes:
            if attempts >= max_attempts or len(hop_paths) >= target_count:
                break

            path = find_single_forward_path_by_access(
                graph, start_node, hop_length, edge_info, nodes, path_signatures, need_access
            )

            if path:
                hop_paths.append(path)

            attempts += 1

        forward_paths.extend(hop_paths)
        print(f"  {hop_length}-hop 路径 ({access_type}): 生成 {len(hop_paths)} 条 (目标: {target_count})")

    return forward_paths

def find_single_forward_path(graph, start_node, hop_length, edge_info, nodes, path_signatures):
    """查找单个正向路径，确保路径内节点和边的唯一性"""
    # 使用DFS查找路径
    stack = [(start_node, [], set())]

    while stack:
        current_node, path, visited = stack.pop()

        if len(path) >= hop_length:
            continue

        # 随机打乱邻居顺序以增加多样性
        neighbors = graph[current_node]
        random.shuffle(neighbors)

        for neighbor, edge_type, mapping_type in neighbors:
            # 只使用有效映射类型
            if mapping_type not in ['1-1', 'n-1']:
                continue

            # 确保路径内节点唯一性：检查邻居节点是否已在当前路径中
            if neighbor in visited:
                continue

            # 确保路径内边唯一性：检查边是否已在当前路径中
            edge_id = edge_info[(current_node, neighbor, edge_type)]['edge_id']
            if edge_id in [p['edge_id'] for p in path]:
                continue

            new_path = path + [{
                'edge_id': edge_id,
                'source_id': current_node,
                'target_id': neighbor,
                'edge_type': edge_type,
                'mapping_type': mapping_type
            }]
            new_visited = visited | {current_node, neighbor}

            # 最终检查：确保新路径中所有节点都是唯一的，且没有走回头路
            all_nodes_in_new_path = set()
            for edge in new_path:
                all_nodes_in_new_path.add(edge['source_id'])
                all_nodes_in_new_path.add(edge['target_id'])

            # 检查1：路径中节点数量应该等于边数量 + 1（确保没有重复节点）
            if len(all_nodes_in_new_path) != len(new_path) + 1:
                continue  # 路径中有重复节点，跳过

            # 检查2：防止走回头路 - 构建节点序列检查是否有重复
            node_sequence = []
            if new_path:
                node_sequence.append(new_path[0]['source_id'])
                for edge in new_path:
                    node_sequence.append(edge['target_id'])

            # 检查是否有任何节点重复出现（走回头路）
            seen_nodes = set()
            for node_id in node_sequence:
                if node_id in seen_nodes:
                    continue  # 发现走回头路，跳过这个路径
                seen_nodes.add(node_id)

            if len(new_path) == hop_length:
                # 检查路径唯一性（边ID组合）
                path_sig = tuple(sorted([p['edge_id'] for p in new_path]))
                if path_sig not in path_signatures:
                    path_signatures.add(path_sig)
                    return new_path
            else:
                stack.append((neighbor, new_path, new_visited))

    return None

def find_single_forward_path_by_access(graph, start_node, hop_length, edge_info, nodes, path_signatures, need_access=True):
    """查找单个正向路径，按need_access分类"""
    # 使用DFS查找路径
    stack = [(start_node, [], set())]

    while stack:
        current_node, path, visited = stack.pop()

        if len(path) >= hop_length:
            continue

        # 随机打乱邻居顺序以增加多样性
        neighbors = graph[current_node]
        random.shuffle(neighbors)

        for neighbor, edge_type, mapping_type in neighbors:
            # 只使用有效映射类型
            if mapping_type not in ['1-1', 'n-1']:
                continue

            # 确保路径内节点唯一性：检查邻居节点是否已在当前路径中
            if neighbor in visited:
                continue

            # 确保路径内边唯一性：检查边是否已在当前路径中
            edge_id = edge_info[(current_node, neighbor, edge_type)]['edge_id']
            if edge_id in [p['edge_id'] for p in path]:
                continue

            edge_data = edge_info[(current_node, neighbor, edge_type)]
            edge_need_access = edge_data.get('need_access', False)

            new_path = path + [{
                'edge_id': edge_id,
                'source_id': current_node,
                'target_id': neighbor,
                'edge_type': edge_type,
                'mapping_type': mapping_type
            }]
            new_visited = visited | {current_node, neighbor}

            # 最终检查：确保新路径中所有节点都是唯一的，且没有走回头路
            all_nodes_in_new_path = set()
            for edge in new_path:
                all_nodes_in_new_path.add(edge['source_id'])
                all_nodes_in_new_path.add(edge['target_id'])

            # 检查1：路径中节点数量应该等于边数量 + 1（确保没有重复节点）
            if len(all_nodes_in_new_path) != len(new_path) + 1:
                continue  # 路径中有重复节点，跳过

            # 检查2：防止走回头路 - 构建节点序列检查是否有重复
            node_sequence = []
            if new_path:
                node_sequence.append(new_path[0]['source_id'])
                for edge in new_path:
                    node_sequence.append(edge['target_id'])

            # 检查是否有任何节点重复出现（走回头路）
            seen_nodes = set()
            for node_id in node_sequence:
                if node_id in seen_nodes:
                    continue  # 发现走回头路，跳过这个路径
                seen_nodes.add(node_id)

            if len(new_path) == hop_length:
                # 检查路径是否符合need_access要求
                if need_access:
                    # 至少有一个edge的need_access为true
                    has_need_access = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if not has_need_access:
                        continue
                else:
                    # 所有edge的need_access都为false
                    has_any_need_access = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if has_any_need_access:
                        continue

                # 检查路径唯一性（边ID组合）
                path_sig = tuple(sorted([p['edge_id'] for p in new_path]))
                if path_sig not in path_signatures:
                    path_signatures.add(path_sig)
                    return new_path
            else:
                # 对于路径中间步骤，也进行need_access检查以提高效率
                if need_access:
                    # 如果当前路径还没有need_access=true的边，优先选择need_access=true的边
                    has_need_access_so_far = any(edge_info[(edge['source_id'], edge['target_id'], edge['edge_type'])].get('need_access', False) for edge in new_path)
                    if not has_need_access_so_far and not edge_need_access:
                        # 优先跳过没有need_access的边，除非已经找到有need_access的边
                        continue
                else:
                    # 如果需要所有边都没有need_access，跳过有need_access的边
                    if edge_need_access:
                        continue

                stack.append((neighbor, new_path, new_visited))

    return None

def find_reverse_paths(graph, edge_info, nodes, max_hops=6, samples_per_hop=10):
    """查找第二类问题路径（以1-n关系开始）"""
    print("查找第二类问题路径...")

    # 找到所有1-n关系的边
    one_n_edges = [edge for edge in edge_info.values() if edge['mapping_type'] == '1-n']

    reverse_paths = []
    path_signatures = set()

    # 为每个跳数生成路径
    for hop_length in range(1, max_hops + 1):
        print(f"生成 {hop_length}-hop 反向路径...")
        hop_paths = []

        # 尝试不同的1-n边作为起点
        random.shuffle(one_n_edges)

        for one_n_edge in one_n_edges:
            if len(hop_paths) >= samples_per_hop:
                break

            # 以1-n关系的source作为起点
            start_node = one_n_edge['source_id']
            fixed_first_edge = {
                'edge_id': one_n_edge['edge_id'],
                'source_id': one_n_edge['source_id'],
                'target_id': one_n_edge['target_id'],
                'edge_type': one_n_edge['edge_type'],
                'mapping_type': '1-n'
            }

            path = find_single_reverse_path(
                graph, one_n_edge['target_id'], hop_length - 1,
                edge_info, nodes, path_signatures, fixed_first_edge
            )

            if path:
                # 将固定的1-n边添加到路径开头
                full_path = [fixed_first_edge] + path
                hop_paths.append(full_path)

        reverse_paths.extend(hop_paths)

    return reverse_paths

def find_single_reverse_path(graph, start_node, remaining_hops, edge_info, nodes, path_signatures, fixed_edges=None):
    """查找单个反向路径，确保路径内节点和边的唯一性"""
    if remaining_hops <= 0:
        return []

    stack = [(start_node, [], set())]

    while stack:
        current_node, path, visited = stack.pop()

        if len(path) >= remaining_hops:
            continue

        neighbors = graph[current_node]
        random.shuffle(neighbors)

        for neighbor, edge_type, mapping_type in neighbors:
            # 只使用1-1和n-1关系
            if mapping_type not in ['1-1', 'n-1']:
                continue

            # 确保路径内节点唯一性：检查邻居节点是否已在当前路径中
            if neighbor in visited:
                continue

            # 确保路径内边唯一性：检查边是否已在当前路径中
            edge_id = edge_info[(current_node, neighbor, edge_type)]['edge_id']
            if edge_id in [p['edge_id'] for p in path]:
                continue

            new_path = path + [{
                'edge_id': edge_id,
                'source_id': current_node,
                'target_id': neighbor,
                'edge_type': edge_type,
                'mapping_type': mapping_type
            }]
            new_visited = visited | {current_node, neighbor}

            # 最终检查：确保新路径中所有节点都是唯一的，且没有走回头路
            all_nodes_in_new_path = set()
            for edge in new_path:
                all_nodes_in_new_path.add(edge['source_id'])
                all_nodes_in_new_path.add(edge['target_id'])

            # 检查1：路径中节点数量应该等于边数量 + 1（确保没有重复节点）
            if len(all_nodes_in_new_path) != len(new_path) + 1:
                continue  # 路径中有重复节点，跳过

            # 检查2：防止走回头路 - 构建节点序列检查是否有重复
            node_sequence = []
            if new_path:
                node_sequence.append(new_path[0]['source_id'])
                for edge in new_path:
                    node_sequence.append(edge['target_id'])

            # 检查是否有任何节点重复出现（走回头路）
            seen_nodes = set()
            for node_id in node_sequence:
                if node_id in seen_nodes:
                    continue  # 发现走回头路，跳过这个路径
                seen_nodes.add(node_id)

            if len(new_path) == remaining_hops:
                # 检查路径唯一性
                all_edges = new_path
                if fixed_edges:
                    all_edges = [fixed_edges] + new_path
                path_sig = tuple(sorted([p['edge_id'] for p in all_edges]))
                if path_sig not in path_signatures:
                    path_signatures.add(path_sig)
                    return new_path
            else:
                stack.append((neighbor, new_path, new_visited))

    return None

def generate_path_description(path, nodes, all_edges_by_edge_id=None):
    """生成具体的路径描述"""
    if not path:
        return ""

    descriptions = []
    for i, edge in enumerate(path):
        source_name = get_node_name(edge['source_id'], nodes)
        target_name = get_node_name(edge['target_id'], nodes)

        # 获取edge的need_access信息
        edge_type_display = edge['edge_type']
        if all_edges_by_edge_id and edge['edge_id'] in all_edges_by_edge_id:
            edge_data = all_edges_by_edge_id[edge['edge_id']]
            if edge_data.get('need_access', False):
                edge_type_display = f"{edge['edge_type']}(need_access)"

        if i == 0:
            descriptions.append(f"{source_name} -> {edge_type_display} -> {target_name}")
        else:
            # 避免重复，直接使用当前边的source和target
            descriptions.append(f"{edge_type_display} -> {target_name}")

    return " -> ".join(descriptions)

def format_path_data(path, nodes, graph, edge_info, target_types, all_edges_by_edge_id, path_type="forward", path_number=1):
    """格式化路径数据"""
    if not path:
        return None

    hop_length = len(path)
    path_id = f"{path_type}_{hop_length}_hop_{path_number}"

    path_edges = [edge['edge_id'] for edge in path]
    path_description = generate_path_description(path, nodes, all_edges_by_edge_id)
    answer = get_node_name(path[-1]['target_id'], nodes)
    answer_type = target_types[path[-1]['target_id']]

    # 计算access_count (need_access=True的edges数量)
    access_count = 0
    has_need_access = False
    for edge in path:
        if edge['edge_id'] in all_edges_by_edge_id:
            if all_edges_by_edge_id[edge['edge_id']].get('need_access', False):
                access_count += 1
                has_need_access = True

    formatted_path = {
        'path_id': path_id,
        'path_type': f"{hop_length}-hop",
        'path_edges': path_edges,
        'path_description': path_description,
        'answer': answer,
        'answer_type': answer_type,
        'path_length': hop_length,
        'access_count': access_count,
        'need_access': has_need_access,  # 新增字段：路径是否包含need_access=true的边
        'detailed_path': [
            {
                'edge_id': edge['edge_id'],
                'source_id': edge['source_id'],
                'source_name': get_node_name(edge['source_id'], nodes),
                'source_type': all_edges_by_edge_id[edge['edge_id']]['source_type'],
                'target_id': edge['target_id'],
                'target_name': get_node_name(edge['target_id'], nodes),
                'target_type': all_edges_by_edge_id[edge['edge_id']]['target_type'],
                'edge_type': edge['edge_type'],
                'mapping_type': edge['mapping_type'],
                'need_access': all_edges_by_edge_id[edge['edge_id']].get('need_access', False)
            }
            for edge in path
        ]
    }

    # 为第二类问题添加约束信息
    if path_type == "reverse" and path:
        first_edge = path[0]
        if first_edge['mapping_type'] == '1-n':
            # 查找约束条件
            constraint = find_constraint_for_entity(first_edge['target_id'], nodes, graph, edge_info)
            if constraint:
                formatted_path['constraint_required'] = constraint
                formatted_path['1-n_relationship'] = {
                    'edge_type': first_edge['edge_type'],
                    'source': first_edge['source_id'],
                    'target': first_edge['target_id'],
                    'source_name': get_node_name(first_edge['source_id'], nodes),
                    'target_name': get_node_name(first_edge['target_id'], nodes)
                }

    return formatted_path

def verify_constraint_uniqueness(entity_id, constraint_conditions, nodes, graph, edge_info):
    """验证约束条件是否能让实体唯一"""
    # 构建查询条件
    matching_entities = []

    # 遍历所有节点，找到符合条件的实体
    for node_id, node in nodes.items():
        if node_id == entity_id:
            continue  # 跳过目标实体本身

        # 检查所有约束条件是否满足
        all_conditions_met = True

        for condition in constraint_conditions:
            attr_name = condition['constraint_attribute']
            attr_value = condition['constraint_value']

            # 检查节点是否有这个属性
            if attr_name not in node['attributes']:
                all_conditions_met = False
                break

            # 检查属性值是否匹配
            node_attr = node['attributes'][attr_name]
            if isinstance(node_attr, dict) and 'name' in node_attr:
                if node_attr['name'] != attr_value:
                    all_conditions_met = False
                    break
            else:
                if node_attr != attr_value:
                    all_conditions_met = False
                    break

        if all_conditions_met:
            matching_entities.append(node_id)

    # 如果没有其他实体满足条件，则约束条件有效
    return len(matching_entities) == 0

def find_constraint_for_entity(entity_id, nodes, graph, edge_info, one_n_source_value=None):
    """为实体查找合适的约束条件"""
    if entity_id not in nodes:
        return None

    entity = nodes[entity_id]
    attributes = entity['attributes']

    # 收集所有可用的约束条件候选
    constraint_candidates = []

    for attr_name, attr_value in attributes.items():
        if isinstance(attr_value, dict) and 'name' in attr_value:
            constraint_candidates.append({
                'constraint_attribute': attr_name,
                'constraint_value': attr_value['name'],
                'constraint_type': attr_value.get('type', 'Unknown'),
                'is_entity_attr': 'node_id' in attr_value
            })

    # 随机打乱候选约束条件
    import random
    random.shuffle(constraint_candidates)

    # 首先尝试单个约束条件
    for candidate in constraint_candidates:
        constraint_conditions = [candidate]

        # 验证这个约束条件是否能让实体唯一
        if verify_constraint_uniqueness(entity_id, constraint_conditions, nodes, graph, edge_info):
            # 找到了有效的单个约束条件
            result = candidate.copy()
            result['selection_reason'] = f"Single constraint '{candidate['constraint_attribute']}' = '{candidate['constraint_value']}' ensures uniqueness"
            return result

    # 如果单个约束条件不够，尝试两个约束条件的组合
    for i in range(len(constraint_candidates)):
        for j in range(i + 1, len(constraint_candidates)):
            constraint_conditions = [constraint_candidates[i], constraint_candidates[j]]

            if verify_constraint_uniqueness(entity_id, constraint_conditions, nodes, graph, edge_info):
                result = {
                    'constraint_attribute': constraint_candidates[i]['constraint_attribute'],
                    'constraint_value': constraint_candidates[i]['constraint_value'],
                    'constraint_type': constraint_candidates[i]['constraint_type'],
                    'additional_constraint': {
                        'constraint_attribute': constraint_candidates[j]['constraint_attribute'],
                        'constraint_value': constraint_candidates[j]['constraint_value'],
                        'constraint_type': constraint_candidates[j]['constraint_type']
                    },
                    'selection_reason': f"Combined constraints '{constraint_candidates[i]['constraint_attribute']}' and '{constraint_candidates[j]['constraint_attribute']}' ensure uniqueness"
                }
                return result

    # 如果两个约束条件还不够，尝试三个约束条件的组合
    for i in range(len(constraint_candidates)):
        for j in range(i + 1, len(constraint_candidates)):
            for k in range(j + 1, len(constraint_candidates)):
                constraint_conditions = [constraint_candidates[i], constraint_candidates[j], constraint_candidates[k]]

                if verify_constraint_uniqueness(entity_id, constraint_conditions, nodes, graph, edge_info):
                    result = {
                        'constraint_attribute': constraint_candidates[i]['constraint_attribute'],
                        'constraint_value': constraint_candidates[i]['constraint_value'],
                        'constraint_type': constraint_candidates[i]['constraint_type'],
                        'additional_constraints': [
                            {
                                'constraint_attribute': constraint_candidates[j]['constraint_attribute'],
                                'constraint_value': constraint_candidates[j]['constraint_value'],
                                'constraint_type': constraint_candidates[j]['constraint_type']
                            },
                            {
                                'constraint_attribute': constraint_candidates[k]['constraint_attribute'],
                                'constraint_value': constraint_candidates[k]['constraint_value'],
                                'constraint_type': constraint_candidates[k]['constraint_type']
                            }
                        ],
                        'selection_reason': f"Three constraints '{constraint_candidates[i]['constraint_attribute']}', '{constraint_candidates[j]['constraint_attribute']}', and '{constraint_candidates[k]['constraint_attribute']}' ensure uniqueness"
                    }
                    return result

    # 如果还是找不到合适的约束条件，返回None
    return None

def main(random_seed=42):
    """主函数"""
    # 设置随机种子以确保可重复性
    random.seed(random_seed)
    print(f"使用随机种子: {random_seed}")
    print("开始生成改进版路径...")

    # 加载数据
    all_edges = load_edges('outputs/edges/clean_edges_filter_success_under_0.5.jsonl')
    all_edges_by_edge_id = {edge['edge_id']: edge for edge in all_edges}
    all_nodes = {}

    node_files = [
        'outputs/nodes/nodes_countries.jsonl',
        'outputs/nodes/nodes_cities.jsonl',
        'outputs/nodes/nodes_universities.jsonl',
        'outputs/nodes/nodes_companies.jsonl',
        'outputs/nodes/nodes_persons.jsonl'
    ]

    for file_path in node_files:
        if os.path.exists(file_path):
            nodes = load_nodes(file_path)
            all_nodes.update(nodes)

    print(f"加载了 {len(all_edges)} 条边和 {len(all_nodes)} 个节点")

    # 构建图
    graph, edge_info, target_types = build_graph(all_edges)

    # 使用统一配置
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.unified_builder import get_path_generation_config

    path_config = get_path_generation_config()
    need_access_config = path_config["forward_paths_need_access"]
    no_need_access_config = path_config["forward_paths_no_need_access"]
    print(f"need_access路径配置: {need_access_config}")
    print(f"no_need_access路径配置: {no_need_access_config}")

    # 生成第一类问题路径 - 分别生成need_access和no_need_access
    forward_paths_data_need_access = find_forward_paths_by_access(graph, edge_info, all_nodes, samples_per_hop_config=need_access_config, need_access=True)
    forward_paths_data_no_need_access = find_forward_paths_by_access(graph, edge_info, all_nodes, samples_per_hop_config=no_need_access_config, need_access=False)

    # 合并两种类型的路径，但保持随机化
    forward_paths_data = []
    all_hop_lengths = set(list(need_access_config.keys()) + list(no_need_access_config.keys()))

    for hop_length in sorted([int(hop.split('-')[0]) for hop in all_hop_lengths]):
        hop_key = f"{hop_length}-hop"

        # 获取当前hop长度的两种类型路径
        need_access_paths = [path for path in forward_paths_data_need_access if len(path) == hop_length]
        no_need_access_paths = [path for path in forward_paths_data_no_need_access if len(path) == hop_length]

        # 合并并随机打乱
        combined_paths = need_access_paths + no_need_access_paths
        random.shuffle(combined_paths)

        forward_paths_data.extend(combined_paths)

        print(f"  {hop_length}-hop 路径: {len(need_access_paths)} need_access, {len(no_need_access_paths)} no_need_access (总计: {len(combined_paths)})")

    # 生成第二类问题路径
    # reverse_paths_data = find_reverse_paths(graph, edge_info, all_nodes, max_hops=6, samples_per_hop=10)

    # 格式化路径数据
    forward_paths = []
    path_counters = defaultdict(int)  # 每个hop长度的计数器

    for path in forward_paths_data:
        hop_length = len(path)
        path_counters[hop_length] += 1

        formatted = format_path_data(
            path, all_nodes, graph, edge_info, target_types,
            all_edges_by_edge_id, "forward", path_counters[hop_length]
        )
        if formatted:
            forward_paths.append(formatted)

    # reverse_paths = []
    # for path in reverse_paths_data:
    #     formatted = format_path_data(path, all_nodes, graph, edge_info, target_types, "reverse")
    #     if formatted:
    #         reverse_paths.append(formatted)

    # 保存路径数据
    os.makedirs('outputs/paths', exist_ok=True)

    with open('outputs/paths/forward_paths.jsonl', 'w', encoding='utf-8') as f:
        for path in forward_paths:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # with open('outputs/paths/reverse_qa_paths.jsonl', 'w', encoding='utf-8') as f:
    #     for path in reverse_paths:
    #         f.write(json.dumps(path, ensure_ascii=False) + '\n')

    # 生成摘要
    summary = {
        'total_analysis': {
            'total_forward_paths': len(forward_paths),
            # 'total_reverse_paths': len(reverse_paths),
            'total_edges': len(all_edges),
            'total_nodes': len(all_nodes)
        },
        'forward_paths_by_hop': defaultdict(int),
        # 'reverse_paths_by_hop': defaultdict(int),
        'path_uniqueness': {
            'forward_unique_edges': len(set(edge for path in forward_paths for edge in path['path_edges']))
            # 'reverse_unique_edges': len(set(edge for path in reverse_paths for edge in path['path_edges']))
        }
    }

    for path in forward_paths:
        summary['forward_paths_by_hop'][path['path_length']] += 1

    # for path in reverse_paths:
    #     summary['reverse_paths_by_hop'][path['path_length']] += 1

    summary['forward_paths_by_hop'] = dict(summary['forward_paths_by_hop'])
    # summary['reverse_paths_by_hop'] = dict(summary['reverse_paths_by_hop'])

    with open('outputs/paths/forward_paths_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("路径生成完成!")
    print(f"第一类问题路径: {len(forward_paths)} 条")
    # print(f"第二类问题路径: {len(reverse_paths)} 条")
    print(f"正向路径分布: {summary['forward_paths_by_hop']}")
    # print(f"反向路径分布: {summary['reverse_paths_by_hop']}")

if __name__ == "__main__":
    random_seed = 42
    main(random_seed)