# generate_wiki_json_multiprocess_robust_json_output.py
# 完整、独立的脚本

import sys
import os
import json
import random
import time
from tqdm import tqdm
from typing import Dict, Set, Optional, List, Any
from functools import partial
import multiprocessing

# ----------------------------
# --- 1. 项目路径与模块导入 ---
# ----------------------------

# 项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from utils.llm_utils_ds import call_model
from prompts.wiki_prompt import HTML_WIKI_PROMPT_TEMPLATE, PLAIN_TEXT_WIKI_PROMPT_TEMPLATE

# ---------------------
# --- 2. 配置区 ---
# ---------------------
    
NODES_DIR = os.path.join(project_root, 'outputs', 'nodes')
WIKI_OUTPUT_DIR = os.path.join(project_root, 'outputs', 'wiki')
WEBSITE_TEMPLATE_DIR = os.path.join(project_root, 'website_template')
API_SLEEP_INTERVAL = 0.1  # 每个worker进程调用API后的休眠时间
NODES_TO_PROCESS_PER_TYPE = None  # 设置为None以处理所有节点, 或设置为数字以限制数量
NUM_WORKERS = 10  # 并行工作进程数，可根据API速率限制和机器性能调整

# --------------------------
# --- 3. 文件操作工具函数 ---
# --------------------------

def load_jsonl(path: str) -> List[dict]:
    """从.jsonl文件中加载数据"""
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    except FileNotFoundError:
        pass  # 文件不存在是正常情况，返回空列表
    return dataset

def save_json(data: Any, save_path: str):
    """将数据以格式化的JSON格式保存到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Successfully saved JSON to: {save_path}')

# ------------------------
# --- 4. 核心辅助函数 ---
# ------------------------

def load_all_nodes(nodes_dir: str) -> Dict[str, List[dict]]:
    """加载所有节点文件，并按类型组织"""
    all_nodes_by_type = {}
    if not os.path.isdir(nodes_dir):
        print(f"Error: Input directory not found at '{nodes_dir}'.")
        # 为模拟运行创建虚拟数据
        print("Creating dummy node data for simulation...")
        os.makedirs(nodes_dir, exist_ok=True)
        dummy_nodes = [
            {"node_id": f"person_{i}", "name": f"Person {i}", "node_type": "Person", "attributes": {}, "attributes_in_abstracts": {}} for i in range(1, 21)
        ]
        with open(os.path.join(nodes_dir, 'nodes_Person.jsonl'), 'w') as f:
            for node in dummy_nodes:
                f.write(json.dumps(node) + '\n')

    print("Loading all node data...")
    for filename in os.listdir(nodes_dir):
        if filename.endswith('.jsonl'):
            node_type = filename.replace('nodes_', '').replace('.jsonl', '')
            filepath = os.path.join(nodes_dir, filename)
            all_nodes_by_type[node_type] = load_jsonl(filepath)
    print(f"All nodes loaded. Found types: {list(all_nodes_by_type.keys())}")
    return all_nodes_by_type

def load_random_html_template() -> str:
    """从website_template目录中随机选择一个HTML模板文件。"""
    if not os.path.exists(WEBSITE_TEMPLATE_DIR):
        # 为模拟运行提供默认模板
        if not os.path.exists(WEBSITE_TEMPLATE_DIR): os.makedirs(WEBSITE_TEMPLATE_DIR, exist_ok=True)
        with open(os.path.join(WEBSITE_TEMPLATE_DIR, 'default.html'), 'w') as f: f.write('<html><body>{{content}}</body></html>')
    
    template_files = [f for f in os.listdir(WEBSITE_TEMPLATE_DIR) if f.endswith('.html')]
    if not template_files:
        return "<html><body>{{content}}</body></html>" # Fallback
    
    selected_template = random.choice(template_files)
    template_path = os.path.join(WEBSITE_TEMPLATE_DIR, selected_template)
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def format_facts_for_prompt(node: dict) -> str:
    """为节点的属性格式化字符串，用于LLM Prompt。"""
    facts = [f"- {k}: {v.get('name')}({v.get('type')})" for k, v in node.get('attributes', {}).items()]
    return "\n".join(facts) if facts else "No specific core facts recorded."

def select_random_facts_for_abstract(node: dict) -> tuple[str, list[str]]:
    """选择用于生成摘要的事实，并返回事实字符串和属性名列表。"""
    facts, names = [], []
    for k, v in node.get('attributes_in_abstracts', {}).items():
        facts.append(f"- {k}: {v.get('name')}({v.get('type')})")
        names.append(k)
    return "\n".join(facts) if facts else "No specific facts selected for abstract.", names

def load_existing_entries_from_json(filepath: str) -> Dict[str, dict]:
    """从主JSON文件中加载已存在的条目，并返回一个ID到条目的映射。"""
    if not os.path.exists(filepath):
        return {}
    
    print(f"Found existing output file. Loading entries from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content or content in ['[]', '[']:
                return {}
            # 尝试修复因中断而未闭合的JSON数组
            if content.endswith(','): content = content[:-1]
            if not content.endswith(']'): content += ']'
            data = json.loads(content)
            existing_map = {entry['id']: entry for entry in data}
            print(f"Loaded {len(existing_map)} existing entries from main JSON.")
            return existing_map
    except json.JSONDecodeError:
        print(f"  >> WARNING: Could not parse '{filepath}'. It might be corrupted. Considering it empty.")
        return {}

def merge_and_save(main_filepath: str, temp_filepath: str):
    """合并主JSON文件和临时JSONL文件，然后保存到主文件并删除临时文件。"""
    if not os.path.exists(temp_filepath):
        print("No temporary file found to merge.")
        return

    print("Merging temporary results into main JSON file...")
    # 1. 加载主文件中的现有数据
    existing_entries_map = load_existing_entries_from_json(main_filepath)

    # 2. 加载临时文件中的新数据
    new_entries = load_jsonl(temp_filepath)
    
    # 3. 合并数据，新数据会更新或添加到现有数据中
    for entry in new_entries:
        existing_entries_map[entry['id']] = entry
    
    # 4. 将合并后的完整数据列表写入主文件
    final_data = list(existing_entries_map.values())
    save_json(final_data, main_filepath)
    
    # 5. 删除临时文件
    os.remove(temp_filepath)
    print(f"Temporary file '{temp_filepath}' has been merged and removed.")

def extract_and_save_abstracts(wiki_json_filepath: str):
    """从最终的wiki.json文件中提取摘要信息并保存到单独的文件。"""
    print("Extracting abstracts from the final result...")
    try:
        with open(wiki_json_filepath, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)

        abstracts_info = {
            entry['id']: {
                'abstract': entry.get('abstract', ''),
                'abstract_attributes': entry.get('abstract_attributes', [])
            } for entry in wiki_data
        }
        abstracts_filepath = wiki_json_filepath.replace('.json', '_abstracts.json')
        save_json(abstracts_info, abstracts_filepath)
    except Exception as e:
        print(f"Error while saving abstract information: {e}")

# -----------------------------
# --- 5. 多进程 Worker 函数 ---
# -----------------------------

def process_node(node: dict, prompt_template: str) -> Optional[dict]:
    """
    处理单个节点的worker函数。调用API并返回生成的条目字典或None。
    """
    node_id, node_name = node['node_id'], node['name']
    try:
        core_facts = format_facts_for_prompt(node)
        abstract_facts, abstract_attributes = select_random_facts_for_abstract(node)
        html_template = load_random_html_template()
        
        prompt_params = {
            "node_name": node_name,
            "node_type": node['node_type'],
            "core_facts_string": core_facts,
            "abstract_facts": abstract_facts,
            "html_template": html_template
        }
        final_prompt = prompt_template.format(**prompt_params)
        
        wiki_data = call_model(final_prompt, response_format={"type": "json_object"})
        
        if not isinstance(wiki_data, dict):
            print(f"  >> WARNING for {node_name} ({node_id}): Expected a dict, got {type(wiki_data)}. Skipping.")
            return None
        
        entry = {
            "id": node_id,
            "title": node_name,
            "abstract": wiki_data.get("abstract", f"Generation failed for {node_name}: Abstract not found."),
            "content": wiki_data.get("content", f"Generation failed for {node_name}: Content not found."),
            "abstract_attributes": abstract_attributes
        }
        
        time.sleep(API_SLEEP_INTERVAL)
        return entry
        
    except Exception as e:
        print(f"  >> UNEXPECTED ERROR in worker for {node_name} ({node_id}): {e}")
        return None

# --------------------------
# --- 6. 主执行流程 ---
# --------------------------

if __name__ == "__main__":
    # 在Windows和macOS上，使用'fork'可能不稳定，'spawn'更安全。
    # 'fork'在Linux上是默认且高效的。这里强制使用'fork'以保持与原始需求的兼容性，
    # 但如果遇到问题，可以尝试注释掉这行。
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)

    # 1. 定义文件路径
    os.makedirs(WIKI_OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(WIKI_OUTPUT_DIR, 'wiki.json')
    temp_output_filepath = os.path.join(WIKI_OUTPUT_DIR, 'wiki_temp.jsonl')  # 临时文件

    # 2. 启动时自愈：检查并合并上次未完成的临时文件
    if os.path.exists(temp_output_filepath):
        print("Found a temporary file from a previous incomplete run. Merging it first...")
        merge_and_save(output_filepath, temp_output_filepath)
    
    # 3. 加载所有节点和现有条目ID
    all_nodes_by_type = load_all_nodes(NODES_DIR)
    existing_entries = load_existing_entries_from_json(output_filepath)
    existing_ids = set(existing_entries.keys())

    # 4. 准备需要处理的任务列表
    tasks = []
    print("\nPreparing tasks for generation...")
    for node_type, nodes in all_nodes_by_type.items():
        nodes_for_processing = nodes[:NODES_TO_PROCESS_PER_TYPE] if NODES_TO_PROCESS_PER_TYPE else nodes
        tasks.extend([node for node in nodes_for_processing if node['node_id'] not in existing_ids])

    if not tasks:
        print("No new nodes to process. All entries seem to be up to date.")
        if os.path.exists(output_filepath):
             extract_and_save_abstracts(output_filepath)
        sys.exit(0)
    
    print(f"Found {len(tasks)} new nodes to process.")
    print(f"Starting generation with {NUM_WORKERS} worker processes.")
    print(f"New results will be temporarily written to '{temp_output_filepath}'")

    # 5. 使用多进程池处理任务，并将结果实时写入临时.jsonl文件
    worker_func = partial(process_node, prompt_template=HTML_WIKI_PROMPT_TEMPLATE)
    new_entries_count = 0

    try:
        # 使用'a'模式（追加）打开临时文件
        with open(temp_output_filepath, 'a', encoding='utf-8') as f_temp, \
             multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            
            results_iterator = pool.imap_unordered(worker_func, tasks)
            
            progress_bar = tqdm(results_iterator, total=len(tasks), desc="Generating entries")
            for result in progress_bar:
                if result:
                    f_temp.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_temp.flush()  # 强制写入磁盘
                    new_entries_count += 1
                    progress_bar.set_postfix_str(f"New entries: {new_entries_count}", refresh=True)

        print(f"\nGeneration phase complete. Wrote {new_entries_count} new entries to the temporary file.")

        # 6. 所有任务成功完成后，执行最终的合并操作
        merge_and_save(output_filepath, temp_output_filepath)
        
        # 7. 最后，从最终的.json文件提取摘要
        extract_and_save_abstracts(output_filepath)

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nProcess interrupted by user (Ctrl+C).")
        else:
            print(f"\nAn unexpected error occurred during the process: {e}")
        
        print(f"Partial results for this run are safely stored in '{temp_output_filepath}'.")
        print("You can re-run the script later to automatically merge them and continue.")
        sys.exit(1)

    print("\nAll tasks completed successfully.")