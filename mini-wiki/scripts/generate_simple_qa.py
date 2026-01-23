import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.qa_prompts import *
from utils.llm_utils_ds import call_model
from tqdm import tqdm 
from scripts.unified_builder import get_simple_qa_templates_config


INPUT_FILE = 'outputs/paths/forward_paths.jsonl'
OUTPUT_DIR = 'outputs/qa/'
OUTPUT_FILE = 'outputs/qa/forward_qa.jsonl'  # 保持向后兼容

def load_existing_qa_for_hop(hop_count, output_dir):
    """
    加载特定hop的已存在QA记录，实现断点续传
    过滤掉question为"API Error"的条目，这些需要重新生成
    """
    output_file = os.path.join(output_dir, f"{hop_count}-hop.jsonl")

    if not os.path.exists(output_file):
        return set(), [], {}

    existing_qa = set()
    existing_entries = []
    api_error_entries = {}  # 存储需要overwrite的API Error条目 {path_id: line_number}

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                if line.strip():
                    qa_entry = json.loads(line.strip())
                    path_id = qa_entry.get('path_id')
                    question = qa_entry.get('question')

                    # 检查question是否为"API Error"，如果是则记录行号用于overwrite
                    if path_id and question != "API Error":
                        existing_qa.add(path_id)
                        existing_entries.append(qa_entry)
                    elif question == "API Error":
                        api_error_entries[path_id] = line_num

        print(f"{hop_count}-hop: 已加载 {len(existing_qa)} 个有效QA记录")
        if api_error_entries:
            print(f"{hop_count}-hop: 发现 {len(api_error_entries)} 个API Error条目，将重新生成并overwrite")

    except Exception as e:
        print(f"警告: 读取 {output_file} 时出错: {e}")
        return set(), [], {}

    return existing_qa, existing_entries, api_error_entries

def generate_questions_for_hop_worker(hop_count, paths, templates, lock):
    """
    单个hop的问题生成工作函数（线程安全）
    """
    results = []

    # 获取已存在的QA记录
    existing_qa, existing_entries, api_error_entries = load_existing_qa_for_hop(hop_count, OUTPUT_DIR)

    # 过滤掉已处理的路径，但保留需要重新生成的API Error路径
    unprocessed_paths = []
    for path in paths:
        path_id = path.get('path_id')
        if path_id not in existing_qa:
            unprocessed_paths.append(path)
        elif path_id in api_error_entries:
            # 这是需要重新生成的API Error路径，也加入处理队列
            unprocessed_paths.append(path)

    if not unprocessed_paths:
        print(f"{hop_count}-hop: 所有路径已处理完成")
        return existing_entries

    print(f"{hop_count}-hop: 开始处理 {len(unprocessed_paths)} 个未处理的路径 (跳过 {len(paths) - len(unprocessed_paths)} 个已处理路径)")

    output_file = os.path.join(OUTPUT_DIR, f"{hop_count}-hop.jsonl")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 线程安全的文件写入函数
    def write_result(result, overwrite_line=None):
        with lock:
            if overwrite_line is not None:
                # overwrite模式：读取整个文件，修改指定行，写回
                with open(output_file, 'r', encoding='utf-8') as f_in:
                    lines = f_in.readlines()

                # 确保行号存在
                if overwrite_line < len(lines):
                    lines[overwrite_line] = json.dumps(result, ensure_ascii=False) + '\n'

                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(lines)
                        f_out.flush()
                    print(f"{hop_count}-hop: Overwrote API Error entry for path {result.get('path_id')} at line {overwrite_line}")
                else:
                    # 行号超出范围，fallback到append模式
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
            else:
                # append模式：正常追加新记录
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()

    # 如果有已存在的记录，先写入它们（如果文件不存在）
    if existing_entries and not os.path.exists(output_file):
        with lock:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for entry in existing_entries:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f_out.flush()

    # 生成新问题
    for i, path_data in enumerate(tqdm(unprocessed_paths, desc=f"Generating {hop_count}-hop", position=hop_count)):
        template_func = templates[i] if i < len(templates) else simple_qa_prompt_all_template
        template_name = next((name for name, func in globals().items() if func == template_func), "unknown")

        question_text, used_template_name = generate_question_for_path(path_data, template_func, template_name)

        if path_data.get("answer_type") in ["Area", 'Company Count', "Employee Count", "Franchise Revenue", "GDP", "Population Count", 'Research Patent Count', 'Research Project Count', "Research Publication Count", "Student Count"]:
            try:
                if int(path_data.get("answer")) >= 10000:
                    # 五位数数字，需要每三位用逗号分隔
                    answer = "{:,}".format(int(path_data.get("answer")))
                else:
                    answer = path_data.get("answer")
            except Exception:
                answer = path_data.get("answer")

        else:
            answer = path_data.get("answer")

        if question_text:
            result = {
                "path_id": path_data.get("path_id"),
                "path_type": path_data.get("path_type"),
                "path_length": path_data.get("path_length"),
                "question": question_text.strip(),
                "answer": answer,
                "answer_type": path_data.get("answer_type"),
                "path_description": path_data.get("path_description"),
                "path_edges": path_data.get("path_edges"),
                "detailed_path": path_data.get("detailed_path"),
                "template_used": used_template_name
            }

            # 检查是否需要overwrite（之前是API Error的条目）
            path_id = path_data.get("path_id")
            overwrite_line = api_error_entries.get(path_id)

            # 立即写入文件（线程安全）
            write_result(result, overwrite_line)
            results.append(result)

    print(f"{hop_count}-hop: 完成生成 {len(results)} 个新问题，总计 {len(existing_entries) + len(results)} 个问题")

    # 返回合并后的结果
    return existing_entries + results

def generate_questions_for_hop_worker_with_progress(hop_count, paths, templates, lock, overall_progress, update_progress_func):
    """
    带有总进度条更新的单个hop问题生成工作函数（线程安全）
    """
    results = []

    # 获取已存在的QA记录
    existing_qa, existing_entries, api_error_entries = load_existing_qa_for_hop(hop_count, OUTPUT_DIR)

    # 过滤掉已处理的路径，但保留需要重新生成的API Error路径
    unprocessed_paths = []
    for path in paths:
        path_id = path.get('path_id')
        if path_id not in existing_qa:
            unprocessed_paths.append(path)
        elif path_id in api_error_entries:
            # 这是需要重新生成的API Error路径，也加入处理队列
            unprocessed_paths.append(path)

    if not unprocessed_paths:
        print(f"{hop_count}-hop: 所有路径已处理完成")
        return existing_entries

    print(f"{hop_count}-hop: 开始处理 {len(unprocessed_paths)} 个未处理的路径 (跳过 {len(paths) - len(unprocessed_paths)} 个已处理路径)")

    output_file = os.path.join(OUTPUT_DIR, f"{hop_count}-hop.jsonl")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 线程安全的文件写入函数
    def write_result(result, overwrite_line=None):
        with lock:
            if overwrite_line is not None:
                # overwrite模式：读取整个文件，修改指定行，写回
                with open(output_file, 'r', encoding='utf-8') as f_in:
                    lines = f_in.readlines()

                # 确保行号存在
                if overwrite_line < len(lines):
                    lines[overwrite_line] = json.dumps(result, ensure_ascii=False) + '\n'

                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(lines)
                        f_out.flush()
                    print(f"{hop_count}-hop: Overwrote API Error entry for path {result.get('path_id')} at line {overwrite_line}")
                else:
                    # 行号超出范围，fallback到append模式
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
            else:
                # append模式：正常追加新记录
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()

    # 如果有已存在的记录，先写入它们（如果文件不存在）
    if existing_entries and not os.path.exists(output_file):
        with lock:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for entry in existing_entries:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f_out.flush()

    # 生成新问题（不使用单独的tqdm，而是更新总进度条）
    for i, path_data in enumerate(unprocessed_paths):
        template_func = templates[i] if i < len(templates) else simple_qa_prompt_all_template
        template_name = next((name for name, func in globals().items() if func == template_func), "unknown")

        question_text, used_template_name = generate_question_for_path(path_data, template_func, template_name)

        if path_data.get("answer_type") in ["Area", 'Company Count', "Employee Count", "Franchise Revenue", "GDP", "Population Count", 'Research Patent Count', 'Research Project Count', "Research Publication Count", "Student Count"]:
            try:
                if int(path_data.get("answer")) >= 10000:
                    # 五位数数字，需要每三位用逗号分隔
                    answer = "{:,}".format(int(path_data.get("answer")))
                else:
                    answer = path_data.get("answer")
            except Exception:
                answer = path_data.get("answer")

        else:
            answer = path_data.get("answer")

        if question_text:
            result = {
                "path_id": path_data.get("path_id"),
                "path_type": path_data.get("path_type"),
                "path_length": path_data.get("path_length"),
                "question": question_text.strip(),
                "answer": answer,
                "answer_type": path_data.get("answer_type"),
                "path_description": path_data.get("path_description"),
                "path_edges": path_data.get("path_edges"),
                "detailed_path": path_data.get("detailed_path"),
                "template_used": used_template_name
            }

            # 检查是否需要overwrite（之前是API Error的条目）
            path_id = path_data.get("path_id")
            overwrite_line = api_error_entries.get(path_id)

            # 立即写入文件（线程安全）
            write_result(result, overwrite_line)
            results.append(result)

        # 更新总进度条
        update_progress_func(1)
        overall_progress.update(1)
        overall_progress.set_postfix_str(f"Current: {hop_count}-hop")

    print(f"{hop_count}-hop: 完成生成 {len(results)} 个新问题，总计 {len(existing_entries) + len(results)} 个问题")

    # 返回合并后的结果
    return existing_entries + results

def generate_weighted_templates(paths_by_hop, templates_config):
    """
    为每个hop生成按权重分配的模板序列，并打乱顺序
    """
    result = {}

    for hop_count, paths in paths_by_hop.items():
        hop_key = f"{hop_count}-hop"
        if hop_key not in templates_config:
            # 如果没有配置，使用默认模板
            result[hop_count] = [simple_qa_prompt_all_template] * len(paths)
            continue

        config = templates_config[hop_key]

        # 计算每个模板的数量
        template_counts = {}
        remaining = len(paths)

        for template_name, weight in config.items():
            count = int(len(paths) * weight)
            template_counts[template_name] = count
            remaining -= count

        # 分配剩余的数量给权重最高的模板
        if remaining > 0:
            max_weight_template = max(config.items(), key=lambda x: x[1])[0]
            template_counts[max_weight_template] += remaining

        # 生成模板序列
        template_sequence = []
        for template_name, count in template_counts.items():
            template_func = globals()[template_name]
            template_sequence.extend([template_func] * count)

        # 打乱模板顺序
        random.shuffle(template_sequence)

        # 确保长度匹配
        template_sequence = template_sequence[:len(paths)]

        result[hop_count] = template_sequence

    return result

def format_facts_from_path(path_data: dict) -> str:
    """
    Formats the edge details from a path object into a human-readable string.
    """
    facts = []
    intermediate_nodes = []

    # 使用 detailed_path 来获取详细信息
    for i, edge_info in enumerate(path_data.get("detailed_path", [])):
        source_name = edge_info.get("source_name", "Unknown")
        edge_type = edge_info.get("edge_type", "related to")
        target_name = edge_info.get("target_name", "Unknown")
        source_type = edge_info.get("source_type", "Unknown")
        target_type = edge_info.get("target_type", "Unknown")

        # 将下划线关系转换为更自然的形式
        natural_relation = edge_type.replace("_", " ")

        facts.append(f"({source_name}({source_type})) --[{natural_relation}]--> ({target_name}({target_type}))")
        if i == 0:
            start_node = edge_info.get("source_name", "Unknown")
        if i != 0:
            intermediate_nodes.append(source_name)

    return "\n".join(facts), intermediate_nodes, start_node, target_type

def generate_question_for_path(path_data: dict, template_func=None, template_name=None):
    """
    Generates a question for a single path by formatting the prompt and calling the LLM.
    Returns (question_text, template_name)
    """
    path_type = path_data.get("path_type", "N-hop")
    answer_name = path_data.get("answer", "Unknown")
    path_length = path_data.get("path_length", 0)

    # Check if essential data is present
    if not path_data.get("detailed_path") or answer_name == "Unknown":
        print(f"Skipping path {path_data.get('path_id')} due to missing details or answer.")
        return None, None

    facts_string, intermediate_nodes, start_node, answer_type = format_facts_from_path(path_data)

    # Use provided template or select default one
    if template_func is None:
        # Default fallback logic if no template provided
        if path_length == 1:
            template_func = simple_qa_prompt_all_template
            template_name = "simple_qa_prompt_all_template"
        elif path_length == 2:
            template_func = simple_qa_prompt_all_template
            template_name = "simple_qa_prompt_all_template"
        elif path_length == 3:
            template_func = simple_qa_prompt_all_template
            template_name = "simple_qa_prompt_all_template"
        else:
            template_func = simple_qa_prompt_all_template
            template_name = "simple_qa_prompt_all_template"

    prompt = template_func.format(
        path_type=path_type,
        facts_string=facts_string,
        intermediate_nodes=intermediate_nodes,
        start_node=start_node,
        answer_name=answer_name,
        answer_type=answer_type
    )

    # Call the language model
    generated_question = call_model(prompt)

    return generated_question, template_name

def load_existing_qa(output_file_path):
    """Load existing QA entries to avoid duplication."""
    if not os.path.exists(output_file_path):
        return set(), []

    existing_qa = set()
    existing_entries = []
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_entry = json.loads(line.strip())
                    path_id = qa_entry.get('path_id')
                    if path_id:
                        existing_qa.add(path_id)
                        existing_entries.append(qa_entry)
        print(f"Loaded {len(existing_qa)} existing QA entries from {output_file_path}")
    except Exception as e:
        print(f"Warning: Could not load existing QA from {output_file_path}: {e}")
        return set(), []

    return existing_qa, existing_entries


def main(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE, test_mode: bool = False, num_samples: int = 3, resume: bool = False, multi_thread: bool = True):
    """
    Main function to read paths, generate questions, and save the results.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            paths = [json.loads(line) for line in f_in]
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Loaded {len(paths)} paths from '{input_file}'. Starting question generation...")

    # 筛选只处理1-6hop的路径
    valid_paths = [p for p in paths if p.get("path_length", 0) >= 1 and p.get("path_length", 0) <= 6]
    print(f"Filtered to {len(valid_paths)} valid paths (1-6 hop).")

    # 检查每个hop的现有QA记录，避免重复生成
    existing_qa_by_hop = {}
    total_existing = 0
    for hop in range(1, 7):
        existing_qa, existing_entries, api_error_entries = load_existing_qa_for_hop(hop, os.path.dirname(output_file))
        existing_qa_by_hop[hop] = existing_qa
        total_existing += len(existing_entries)

    if total_existing > 0:
        print(f"Found existing QA entries: {total_existing} total across all hop files")

        # 过滤掉已存在的entries
        filtered_paths = []
        for path in valid_paths:
            hop_count = path.get("path_length", 0)
            if path.get('path_id') not in existing_qa_by_hop.get(hop_count, set()):
                filtered_paths.append(path)

        print(f"Filtered to {len(filtered_paths)} new paths to generate (skipped {len(valid_paths) - len(filtered_paths)} already processed)")
        valid_paths = filtered_paths
    else:
        print("No existing QA entries found. Creating new hop files.")

    # 按hop类型分组
    paths_by_hop = {}
    for path in valid_paths:
        hop_count = path.get("path_length", 0)
        if hop_count not in paths_by_hop:
            paths_by_hop[hop_count] = []
        paths_by_hop[hop_count].append(path)

    print(f"Path distribution by hop count:")
    for hop in sorted(paths_by_hop.keys()):
        print(f"  {hop}-hop: {len(paths_by_hop[hop])} paths")

    # 获取模板配置并生成加权模板序列
    templates_config = get_simple_qa_templates_config()
    print(f"Template configuration: {templates_config}")

    # 生成加权模板序列
    templates_by_hop = generate_weighted_templates(paths_by_hop, templates_config)

    print("Template allocation by hop:")
    for hop, templates in templates_by_hop.items():
        template_counts = {}
        for template in templates:
            template_name = next((name for name, func in globals().items() if func == template), "unknown")
            template_counts[template_name] = template_counts.get(template_name, 0) + 1
        print(f"  {hop}-hop: {template_counts}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 根据模式选择路径
    selected_paths = []
    if test_mode:
        # 测试模式：从每个hop类型中随机抽取指定数量的路径
        for hop in sorted(paths_by_hop.keys()):
            available_paths = paths_by_hop[hop]
            sample_size = min(num_samples, len(available_paths))
            sampled_paths = random.sample(available_paths, sample_size)
            selected_paths.extend(sampled_paths)
            print(f"Selected {sample_size} paths for {hop}-hop test")
    else:
        # 生产模式：选择所有路径
        selected_paths = valid_paths
        print(f"Selected all {len(selected_paths)} paths for generation")

    print(f"\nTotal selected paths: {len(selected_paths)}")

    # 如果没有新路径需要生成，直接返回
    if len(selected_paths) == 0:
        print("No new paths to generate. All paths have been processed.")
        return

    # 按hop重新组织选中的路径并分配模板
    selected_paths_by_hop = {}
    for path in selected_paths:
        hop_count = path.get("path_length", 0)
        if hop_count not in selected_paths_by_hop:
            selected_paths_by_hop[hop_count] = []
        selected_paths_by_hop[hop_count].append(path)

    # 为每个hop分配对应的模板序列
    templates_for_selected_paths = {}
    for hop_count, paths in selected_paths_by_hop.items():
        if hop_count in templates_by_hop:
            # 使用预先生成的模板序列的前N个
            templates_for_selected_paths[hop_count] = templates_by_hop[hop_count][:len(paths)]
        else:
            # 如果没有配置，使用默认模板
            templates_for_selected_paths[hop_count] = [simple_qa_prompt_all_template] * len(paths)

    print(f"\nTotal selected paths by hop:")
    for hop_count in sorted(selected_paths_by_hop.keys()):
        print(f"  {hop_count}-hop: {len(selected_paths_by_hop[hop_count])} paths")

    # 计算总路径数用于进度条
    total_paths = sum(len(paths) for paths in selected_paths_by_hop.values())

    # 创建共享进度跟踪器
    progress_lock = threading.Lock()
    completed_paths = 0

    def update_progress(completed=1):
        nonlocal completed_paths
        with progress_lock:
            completed_paths += completed
            return completed_paths

    if multi_thread:
        # 多线程模式
        print(f"\nStarting multi-threaded question generation...")
        print("Each hop will be processed in parallel (max 6 threads)")

        # 创建总进度条
        overall_progress = tqdm(total=total_paths, desc="Overall Progress", position=0)

        # 创建线程锁
        file_lock = threading.Lock()

        # 使用线程池并发处理每个hop
        with ThreadPoolExecutor(max_workers=6) as executor:
            # 提交任务
            futures = []
            for hop_count in sorted(selected_paths_by_hop.keys()):
                paths = selected_paths_by_hop[hop_count]
                templates = templates_for_selected_paths[hop_count]

                print(f"Submitting task for {hop_count}-hop ({len(paths)} paths)...")
                future = executor.submit(generate_questions_for_hop_worker_with_progress, hop_count, paths, templates, file_lock, overall_progress, update_progress)
                futures.append((hop_count, future))

            # 等待所有任务完成并收集结果
            all_results = []
            for hop_count, future in futures:
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"{hop_count}-hop: Completed successfully")
                except Exception as e:
                    print(f"{hop_count}-hop: Failed with error: {e}")

        overall_progress.close()
        print(f"\nMulti-threaded question generation complete!")
        print(f"Total questions generated: {len(all_results)}")

    else:
        # 单线程模式
        print(f"\nStarting single-threaded question generation...")

        # 创建总进度条
        overall_progress = tqdm(total=total_paths, desc="Overall Progress", position=0)

        # 按hop顺序处理路径以确保模板分配正确
        for hop_count in sorted(selected_paths_by_hop.keys()):
            paths = selected_paths_by_hop[hop_count]
            templates = templates_for_selected_paths[hop_count]

            print(f"Processing {hop_count}-hop paths with {len(templates)} templates...")

            # 使用worker函数处理单个hop，传入进度条
            results = generate_questions_for_hop_worker_with_progress(hop_count, paths, templates, threading.Lock(), overall_progress, update_progress)
            print(f"{hop_count}-hop: Completed {len(results)} questions")

        overall_progress.close()
        print(f"\nSingle-threaded question generation complete!")

    # 生成汇总报告
    print(f"\n" + "="*60)
    print("汇总报告")
    print("="*60)

    for hop_count in sorted(selected_paths_by_hop.keys()):
        output_file = os.path.join(OUTPUT_DIR, f"{hop_count}-hop.jsonl")
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for line in f if line.strip())
                print(f"{hop_count}-hop: {count} 个问题已保存到 {output_file}")
            except Exception as e:
                print(f"{hop_count}-hop: 读取结果文件时出错: {e}")
        else:
            print(f"{hop_count}-hop: 未生成结果文件")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simple QA for forward paths")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input file path")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file path (main merged file)")
    parser.add_argument("--test", action="store_true", help="Test mode: generate only 10 samples per hop type")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per hop type in test mode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file, skip already generated QAs")
    parser.add_argument("--multi-thread", action="store_true", default=True, help="Use multi-threading (default: True)")
    parser.add_argument("--single-thread", action="store_true", help="Use single-threading instead of multi-threading")

    args = parser.parse_args()

    # 处理线程模式选择
    multi_thread = not args.single_thread  # 默认多线程，除非指定--single-thread

    main(
        input_file=args.input,
        output_file=args.output,
        test_mode=args.test,
        num_samples=args.samples,
        resume=args.resume,
        multi_thread=multi_thread
    )