import os
import json
from typing import Dict, List, Any

# --- 全局配置 ---
NODES_DIR = 'outputs/nodes'
KNOWN_KEYS_FILE = 'outputs/known_attribute_keys.json'

def load_schema(schema_path: str = 'schema_config.json') -> Dict[str, Any]:
    """加载并返回 schema 配置文件。"""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_nodes(node_type: str) -> List[Dict[str, Any]]:
    """加载指定类型的节点文件。如果文件不存在，返回空列表。"""
    file_path = os.path.join(NODES_DIR, f"nodes_{node_type.lower()}.jsonl")
    if not os.path.exists(file_path):
        return []
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            nodes.append(json.loads(line))
    return nodes

def save_nodes(node_type: str, nodes: List[Dict[str, Any]]):
    """将节点列表保存回对应的 .jsonl 文件。"""
    os.makedirs(NODES_DIR, exist_ok=True)
    file_path = os.path.join(NODES_DIR, f"{node_type}.jsonl")
    with open(file_path, 'w', encoding='utf-8') as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + '\n')

# --- 新增的核心功能函数 ---

def update_known_attribute_keys():
    """
    扫描所有节点文件，收集所有用到的属性键，并更新 known_attribute_keys.json 文件。
    """
    print("\n--- Updating known attribute keys ---")
    
    # 尝试加载 schema 来获取所有已定义的 node_type
    try:
        schema = load_schema()
        node_types_to_scan = schema.get('node_types', {}).keys()
    except FileNotFoundError:
        print("Warning: schema_config.json not found. Scanning NODES_DIR for existing .jsonl files.")
        # 如果 schema 不存在，则回退到扫描文件夹中的文件
        if not os.path.exists(NODES_DIR):
            print("Nodes directory not found. Nothing to scan.")
            return
        node_types_to_scan = [f.replace('.jsonl', '') for f in os.listdir(NODES_DIR) if f.endswith('.jsonl')]

    if not node_types_to_scan:
        print("No node types found to scan.")
        return

    known_keys = set()
    
    # 遍历所有 node_type
    for node_type in node_types_to_scan:
        nodes = load_nodes(node_type)
        for node in nodes:
            # 安全地获取 attributes 字典
            for key in node.get('attributes', {}).keys():
                # 避免重复添加
                if key in known_keys:
                    continue
                else:
                    known_keys.add(key)
    
    # 读取已有的 keys，进行合并，而不是完全覆盖
    if os.path.exists(KNOWN_KEYS_FILE):
        with open(KNOWN_KEYS_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_keys = set(json.load(f))
                known_keys.update(existing_keys)
            except json.JSONDecodeError:
                print(f"Warning: Could not read existing keys from {KNOWN_KEYS_FILE}. The file will be overwritten.")

    with open(KNOWN_KEYS_FILE, 'w', encoding='utf-8') as f:
        # 转换为列表以便 JSON 序列化，并排序以保持文件稳定性和可读性
        json.dump(sorted(list(known_keys)), f, indent=2, ensure_ascii=False)
        
    print(f"Successfully updated known attribute keys file at: '{KNOWN_KEYS_FILE}'")
    print(f"Total keys found: {len(known_keys)}")