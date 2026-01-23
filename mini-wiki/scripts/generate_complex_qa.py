#!/usr/bin/env python3
"""
Main script to generate complex QA for all types.
Integrates all four QA generation methods with configurable settings.
"""

import json
import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_utils_ds import call_model



def safe_int_conversion(value):
    """Safely convert a value to int, handling various formats including commas."""
    try:
        if isinstance(value, str):
            # Remove commas and other formatting characters
            cleaned = value.replace(',', '').replace('$', '').replace(' ', '')
            return int(cleaned)
        else:
            return int(value)
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert '{value}' to int: {e}")
        raise ValueError(f"Invalid number format: {value}")


def format_scientific_notation(value):
    """Convert numbers to comma-separated format if they have 5 or more digits."""
    try:
        # First, safely convert to int (handles existing commas)
        num = safe_int_conversion(value)

        # Check if it's a number with 5 or more digits (absolute value >= 10000)
        if abs(num) >= 10000:
            # Use comma formatting
            return f"{num:,}"
        else:
            # For numbers with 4 digits or less, return as string without commas
            return str(num)
    except (ValueError, TypeError):
        # If not a number, return as is
        return str(value)
from prompts.qa_prompts import (
    combo_qa_prompt_template,
    parallel_same_numerical_sum_prompt_template,
    parallel_same_numerical_difference_prompt_template,
    parallel_same_entity_compare_prompt_template,
    parallel_same_entity_sum_prompt_template,
    parallel_same_entity_difference_prompt_template
)


def load_node_data():
    """Load node data to get attribute relations."""
    nodes = {}
    node_files = [
        "outputs/nodes/nodes_persons.jsonl",
        "outputs/nodes/nodes_companies.jsonl",
        "outputs/nodes/nodes_cities.jsonl",
        "outputs/nodes/nodes_universities.jsonl",
        "outputs/nodes/nodes_countries.jsonl"
    ]

    for file_path in node_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    node = json.loads(line.strip())
                    nodes[node['node_id']] = node

    return nodes


def load_forward_qa_data(file_path):
    """Load forward QA data and create path_id to QA mapping."""
    path_to_qa = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            qa_entry = json.loads(line.strip())
            path_id = qa_entry['path_id']
            path_to_qa[path_id] = qa_entry
    return path_to_qa


def load_complex_paths_data(file_path):
    """Load complex paths data from jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def combine_questions_simple(sub_questions):
    """Combine two sub-questions into a single complex question (simple concatenation)."""
    if len(sub_questions) != 2:
        raise ValueError(f"Expected exactly 2 sub-questions, got {len(sub_questions)}")

    q1, q2 = sub_questions
    combined = f"Q1: {q1}\nQ2: {q2}\n\nPlease Answer both questions."
    return combined


def get_attribute_relation(node_id, attribute_name, nodes):
    """Get the attribute relation string for a node."""
    if node_id not in nodes:
        return f"Unknown node {node_id}"

    node = nodes[node_id]
    if attribute_name not in node.get('attributes', {}):
        return f"No attribute {attribute_name} for node {node_id}"

    attr_value = node['attributes'][attribute_name]
    return f"{node['name']} -> {attribute_name} -> {attr_value['name']}"


def select_prompt_template(answer_type):
    """Select appropriate prompt template based on answer type."""
    # Randomly choose between sum and difference for all numerical types
    if random.random() < 0.5:
        return parallel_same_numerical_difference_prompt_template, "difference"
    else:
        return parallel_same_numerical_sum_prompt_template, "sum"


def generate_parallel_diff_type_qa(data, forward_qa_mapping, config, output_file):
    """Generate complex QA for parallel different type questions."""
    print(f"Generating parallel diff type QA...")

    # In test mode, use all data but stop after generating 3 successful entries
    sample_data = data if not config['test_mode'] else data
    count = 0
    skipped = 0
    max_test_samples = config.get('num_samples', 3) if config['test_mode'] else float('inf')

    for entry in tqdm(sample_data, desc="Generating parallel diff type QA", unit="entry"):
        # Check if original_path_ids exists
        if "original_path_ids" not in entry or len(entry["original_path_ids"]) != 2:
            print(f"  ✗ Missing original_path_ids for entry {entry.get('path_id', 'unknown')}")
            skipped += 1
            continue

        path_id1, path_id2 = entry["original_path_ids"][0], entry["original_path_ids"][1]

        # Look up sub-QAs from forward_qa mapping
        if path_id1 not in forward_qa_mapping or path_id2 not in forward_qa_mapping:
            print(f"  ✗ Cannot find sub-QAs for paths {path_id1}, {path_id2}")
            skipped += 1
            continue

        qa1 = forward_qa_mapping[path_id1]
        qa2 = forward_qa_mapping[path_id2]

        sub_questions = [qa1["question"], qa2["question"]]
        answers = entry.get("answers", [qa1["answer"], qa2["answer"]])
        answer_types = entry.get("answer_type", [qa1["answer_type"], qa2["answer_type"]])
        if not isinstance(answer_types, list):
            answer_types = [answer_types] * 2

        new_entry = entry.copy()
        new_entry["sub_questions"] = sub_questions
        new_entry["answers"] = answers
        new_entry["answer_types"] = answer_types
        new_entry["answer"] = format_scientific_notation(answers[0]) + ", " + format_scientific_notation(answers[1])
        new_entry["question"] = combine_questions_simple(sub_questions)

        output_file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        output_file.flush()
        count += 1

        # Stop if we've generated enough for test mode
        if config['test_mode'] and count >= max_test_samples:
            print(f"  ✓ Generated {count} entries, stopping early (test mode limit reached)")
            break

    print(f"Generated {count} entries, skipped {skipped} entries")
    return count


def generate_combo_qa(data, forward_qa_mapping, config, output_file):
    """Generate complex QA for combo questions using LLM."""
    print(f"Generating combo QA using LLM...")

    # In test mode, use all data but stop after generating 3 successful entries
    sample_data = data if not config['test_mode'] else data
    count = 0
    skipped = 0
    max_test_samples = config.get('num_samples', 3) if config['test_mode'] else float('inf')

    for entry in tqdm(sample_data, desc="Generating combo QA", unit="entry"):

        # Check if original_path_ids exists
        if "original_path_ids" not in entry or len(entry["original_path_ids"]) != 2:
            print(f"  ✗ Missing original_path_ids for entry {entry.get('path_id', 'unknown')}")
            skipped += 1
            continue

        path_id1, path_id2 = entry["original_path_ids"][0], entry["original_path_ids"][1]

        # Look up sub-QAs from forward_qa mapping
        if path_id1 not in forward_qa_mapping or path_id2 not in forward_qa_mapping:
            print(f"  ✗ Cannot find sub-QAs for paths {path_id1}, {path_id2}")
            skipped += 1
            continue

        qa1 = forward_qa_mapping[path_id1]
        qa2 = forward_qa_mapping[path_id2]

        # Prepare prompt parameters
        sub_question1 = qa1["question"]
        sub_question2 = qa2["question"]
        sub_answer1 = qa1["answer"]
        sub_answer2 = qa2["answer"]
        sub_answer_type1 = qa1["answer_type"]
        sub_answer_type2 = qa2["answer_type"]

        # Format the prompt
        prompt = combo_qa_prompt_template.replace(
            "{sub_question1}", sub_question1
        ).replace(
            "{sub_question2}", sub_question2
        ).replace(
            "{sub_answer1}", str(sub_answer1)
        ).replace(
            "{sub_answer2}", str(sub_answer2)
        ).replace(
            "{sub_answer_type1}", sub_answer_type1
        ).replace(
            "{sub_answer_type2}", sub_answer_type2
        )

        # Call LLM
        llm_response = call_model(prompt)

        if llm_response is None:
            print(f"  ✗ Failed to get LLM response")
            skipped += 1
            continue

        # Create final question format
        final_question = f"Q1: {sub_question1}\nQ2: {llm_response}\n\nPlease solve this step-by-step problem. You must answer Q1 first, then use  Q1's answer to solve Q2. Your final response should only be the answer to Q2."

        # Format sub_answer2 with scientific notation if it's a number
        formatted_answer = format_scientific_notation(sub_answer2)

        # Return new entry
        new_entry = entry.copy()
        new_entry["sub_questions"] = [sub_question1, sub_question2]
        new_entry["answers"] = [sub_answer1, sub_answer2]
        new_entry["answer_types"] = [sub_answer_type1, sub_answer_type2]
        new_entry["rephrased_sub_question2"] = llm_response
        new_entry["question"] = final_question
        new_entry["answer"] = formatted_answer  # Always use subquestion2's answer (formatted)

        output_file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        output_file.flush()
        count += 1
        # tqdm already shows progress, no need for additional progress print

        # Stop if we've generated enough for test mode
        if config['test_mode'] and count >= max_test_samples:
            print(f"  ✓ Generated {count} entries, stopping early (test mode limit reached)")
            break

    print(f"Generated {count} entries, skipped {skipped} entries")
    return count


def generate_parallel_same_entity_qa(data, forward_qa_mapping, config, output_file):
    """Generate complex QA for parallel same type entity questions using rule-based templates."""
    print(f"Generating parallel same type entity QA using rule-based templates...")

    # In test mode, use all data but stop after generating 3 successful entries
    sample_data = data if not config['test_mode'] else data
    count = 0
    skipped = 0
    max_test_samples = config.get('num_samples', 3) if config['test_mode'] else float('inf')

    for entry in tqdm(sample_data, desc="Generating parallel same type entity QA", unit="entry"):

        # Check if original_path_ids exists
        if "original_path_ids" not in entry or len(entry["original_path_ids"]) != 2:
            print(f"  ✗ Missing original_path_ids for entry {entry.get('path_id', 'unknown')}")
            skipped += 1
            continue

        path_id1, path_id2 = entry["original_path_ids"][0], entry["original_path_ids"][1]

        # Look up sub-QAs from forward_qa mapping
        if path_id1 not in forward_qa_mapping or path_id2 not in forward_qa_mapping:
            print(f"  ✗ Cannot find sub-QAs for paths {path_id1}, {path_id2}")
            skipped += 1
            continue

        qa1 = forward_qa_mapping[path_id1]
        qa2 = forward_qa_mapping[path_id2]

        # Get sub-questions
        sub_question1 = qa1["question"]
        sub_question2 = qa2["question"]

        # Check if required fields exist
        required_fields = ["same_numerical_attribute", "same_numerical_attribute_value"]
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            print(f"  ✗ Missing required fields: {missing_fields}")
            skipped += 1
            continue

        # Get numerical attribute information
        attr_name = entry["same_numerical_attribute"][0]  # Both are the same attribute
        value1 = entry["same_numerical_attribute_value"][0]
        value2 = entry["same_numerical_attribute_value"][1]

        # Select operation type randomly
        if random.random() < 0.5:
            operation_type = "sum"
            operation_word = "sum"
            calculation = safe_int_conversion(value1) + safe_int_conversion(value2)
        else:
            operation_type = "difference"
            operation_word = "absolute difference"
            calculation = abs(safe_int_conversion(value1) - safe_int_conversion(value2))

        # Format attribute name for human reading
        attribute_name = attr_name.replace('_', ' ')

        # Create the complete question with Q1, Q2, Q3 format
        q3_question = f"What is the {operation_word} of the {attribute_name} of the answer to Q1 and the {attribute_name} of the answer to Q2?"

        # Complete instruction format
        complete_question = f"""Q1: {sub_question1}
Q2: {sub_question2}
Q3: {q3_question}

Please solve this step-by-step problem. You must answer Q1 and Q2 first, then use those answers to solve Q3. Your final response should only be the numerical answer to Q3."""

        final_answer = str(calculation)

        # Return new entry
        new_entry = entry.copy()
        new_entry["sub_questions"] = [sub_question1, sub_question2]
        new_entry["q3_question"] = q3_question
        new_entry["question"] = complete_question
        new_entry["answer"] = final_answer
        new_entry["numerical_values"] = [value1, value2]
        new_entry["question_type"] = operation_type
        new_entry["attribute_name"] = attribute_name
        output_file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        output_file.flush()
        count += 1

        # Stop if we've generated enough for test mode
        if config['test_mode'] and count >= max_test_samples:
            print(f"  ✓ Generated {count} entries, stopping early (test mode limit reached)")
            break

    print(f"Generated {count} entries, skipped {skipped} entries")
    return count

def generate_parallel_same_numerical_qa(data, forward_qa_mapping, config, output_file):
    """Generate complex QA for parallel same type numerical questions using LLM."""
    print(f"Generating parallel same type numerical QA using LLM...")

    # In test mode, use all data but stop after generating 3 successful entries
    sample_data = data if not config['test_mode'] else data
    count = 0
    skipped = 0
    max_test_samples = config.get('num_samples', 3) if config['test_mode'] else float('inf')

    for entry in tqdm(sample_data, desc="Generating parallel same type numerical QA", unit="entry"):

        # Check if original_path_ids exists
        if "original_path_ids" not in entry or len(entry["original_path_ids"]) != 2:
            print(f"  ✗ Missing original_path_ids for entry {entry.get('path_id', 'unknown')}")
            skipped += 1
            continue

        path_id1, path_id2 = entry["original_path_ids"][0], entry["original_path_ids"][1]

        # Look up sub-QAs from forward_qa mapping
        if path_id1 not in forward_qa_mapping or path_id2 not in forward_qa_mapping:
            print(f"  ✗ Cannot find sub-QAs for paths {path_id1}, {path_id2}")
            skipped += 1
            continue

        qa1 = forward_qa_mapping[path_id1]
        qa2 = forward_qa_mapping[path_id2]

        # Check if answer types are the same
        if qa1["answer_type"] != qa2["answer_type"]:
            print(f"  ✗ Answer types don't match: {qa1['answer_type']} vs {qa2['answer_type']}")
            skipped += 1
            continue

        # Prepare prompt parameters
        sub_question1 = qa1["question"]
        sub_question2 = qa2["question"]
        sub_answer1 = qa1["answer"]
        sub_answer2 = qa2["answer"]
        sub_answer_type1 = qa1["answer_type"]
        sub_answer_type2 = qa2["answer_type"]

        # Select prompt template
        prompt_template, question_type = select_prompt_template(sub_answer_type1)

        # Format the prompt
        prompt = prompt_template.replace(
            "{sub_question1}", sub_question1
        ).replace(
            "{sub_question2}", sub_question2
        ).replace(
            "{sub_answer1}", str(sub_answer1)
        ).replace(
            "{sub_answer2}", str(sub_answer2)
        ).replace(
            "{sub_answer_type1}", sub_answer_type1
        ).replace(
            "{sub_answer_type2}", sub_answer_type2
        )

        # Call LLM with JSON response format
        response_format = {"type": "json_object"}
        llm_response = call_model(prompt, response_format=response_format)

        if llm_response is None:
            print(f"  ✗ Failed to get LLM response")
            continue

        # Extract question and answer from LLM response
        if isinstance(llm_response, dict) and "Question" in llm_response and "Answer" in llm_response:
            final_question = llm_response["Question"]
            final_answer = llm_response["Answer"]
             
            if question_type == "sum":
                ground_truth = safe_int_conversion(sub_answer1) + safe_int_conversion(sub_answer2)
                if final_answer != str(ground_truth):
                    print(f"  ⚠️  WARNING: Incorrect LLM answer detected")
                    print(f"     Question ID: {entry.get('path_id', 'unknown')}")
                    print(f"     Original answers: {sub_answer1}, {sub_answer2}")
                    print(f"     Expected answer: {ground_truth}")
                    print(f"     LLM answer: {final_answer}")
                    print(f"     Correcting answer to: {ground_truth}")
                    final_answer = ground_truth
                    
            elif question_type == "difference":
                ground_truth = abs(safe_int_conversion(sub_answer1) - safe_int_conversion(sub_answer2))
                if final_answer != str(ground_truth):
                    print(f"  ⚠️  WARNING: Incorrect LLM answer detected")
                    print(f"     Question ID: {entry.get('path_id', 'unknown')}")
                    print(f"     Original answers: {sub_answer1}, {sub_answer2}")
                    print(f"     Expected answer: {ground_truth}")
                    print(f"     LLM answer: {final_answer}")
                    print(f"     Correcting answer to: {ground_truth}")
                    final_answer = ground_truth
            
            else:
                print(f"  ✗ Invalid question type: {question_type}")
                skipped += 1
                continue

        else:
            print(f"  ✗ Invalid LLM response format")
            skipped += 1
            continue

        # Return new entry
        new_entry = entry.copy()
        new_entry["sub_questions"] = [sub_question1, sub_question2]
        new_entry["answers"] = [sub_answer1, sub_answer2]
        new_entry["question"] = final_question
        new_entry["answer"] = final_answer
        new_entry["question_type"] = question_type
        output_file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        output_file.flush()
        count += 1
        # tqdm already shows progress, no need for additional progress print

        # Stop if we've generated enough for test mode
        if config['test_mode'] and count >= max_test_samples:
            print(f"  ✓ Generated {count} entries, stopping early (test mode limit reached)")
            break

    print(f"Generated {count} entries, skipped {skipped} entries")
    return count


def save_results(results, output_path):
    """Save results to jsonl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} entries to {output_path}")


def get_default_config():
    """Get default configuration for all QA types."""
    return {
        "parallel_diff_type": {
            "enabled": True,
            "test_mode": False,
            "num_samples": 1,
            "input_file": "outputs/complex_paths/parallel_diff_type_qa_paths.jsonl",
            "forward_qa_file": "outputs/qa/forward_qa.jsonl",
            "output_file": "outputs/qa/parallel_diff_type_qa.jsonl"
        },
        "combo": {
            "enabled": True,
            "test_mode": False,
            "num_samples": 1,
            "input_file": "outputs/complex_paths/combo_qa_paths.jsonl",
            "forward_qa_file": "outputs/qa/forward_qa.jsonl",
            "output_file": "outputs/qa/combo_qa.jsonl"
        },
        "parallel_same_entity": {
            "enabled": True,
            "test_mode": False,
            "num_samples": 1,
            "input_file": "outputs/complex_paths/parallel_same_type_entity_qa_paths.jsonl",
            "forward_qa_file": "outputs/qa/forward_qa.jsonl",
            "output_file": "outputs/qa/parallel_same_type_entity_qa.jsonl"
        },
        "parallel_same_numerical": {
            "enabled": True,
            "test_mode": False,
            "num_samples": 1,
            "input_file": "outputs/complex_paths/parallel_same_type_numerical_qa_paths.jsonl",
            "forward_qa_file": "outputs/qa/forward_qa.jsonl",
            "output_file": "outputs/qa/parallel_same_type_numerical_qa.jsonl"
        }
    }


def load_existing_qa(output_file_path):
    """Load existing QA entries to avoid duplication."""
    if not os.path.exists(output_file_path):
        return set()

    existing_qa = set()
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_entry = json.loads(line.strip())
                    path_id = qa_entry.get('path_id')
                    if path_id:
                        existing_qa.add(path_id)
        print(f"Loaded {len(existing_qa)} existing QA entries from {output_file_path}")
    except Exception as e:
        print(f"Warning: Could not load existing QA from {output_file_path}: {e}")
        return set()

    return existing_qa


def main():
    parser = argparse.ArgumentParser(description="Generate complex QA for all types")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--test-all", action="store_true", help="Run all types in test mode")
    parser.add_argument("--prod-all", action="store_true", help="Run all types in production mode")
    parser.add_argument("--test-only", action="store_true", help="Only generate 3 samples per type for testing")
    parser.add_argument("--types", type=str, nargs="+",
                       choices=["parallel_diff_type", "combo", "parallel_same_entity", "parallel_same_numerical"],
                       help="Specific QA types to generate")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples for test mode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files, skip already generated QAs")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()

    # Override config with command line arguments
    if args.test_all:
        for qa_type in config:
            config[qa_type]["test_mode"] = True
            config[qa_type]["num_samples"] = args.samples
    elif args.prod_all:
        for qa_type in config:
            config[qa_type]["test_mode"] = False
    elif args.test_only:
        for qa_type in config:
            config[qa_type]["test_mode"] = True
            config[qa_type]["num_samples"] = 3

    # Filter types if specified
    if args.types:
        config = {k: v for k, v in config.items() if k in args.types}

    # Load node data (needed for some QA types)
    print("Loading node data...")
    nodes = load_node_data()
    print(f"Loaded {len(nodes)} nodes")

    # Load forward QA data for sub-questions lookup
    print("Loading forward QA data...")
    forward_qa_file = "outputs/qa/forward_qa.jsonl"
    if not os.path.exists(forward_qa_file):
        print(f"Forward QA file {forward_qa_file} does not exist, exiting...")
        return

    forward_qa_mapping = load_forward_qa_data(forward_qa_file)
    print(f"Loaded {len(forward_qa_mapping)} forward QA entries")

    # Generate QA for each type
    total_results = {}
    for qa_type, qa_config in config.items():
        if not qa_config["enabled"]:
            continue

        print(f"\n{'='*60}")
        print(f"PROCESSING: {qa_type}")
        print(f"{'='*60}")

        input_file = qa_config["input_file"]
        output_file_path = qa_config["output_file"]

        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist, skipping...")
            continue

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Add nodes to config for entity processing
        qa_config_with_nodes = {**qa_config, 'nodes': nodes}

        # Check if output file exists and load existing QAs
        existing_qa = load_existing_qa(output_file_path)

        # In test mode, check if we already have 3 entries
        if qa_config_with_nodes.get("test_mode", False) and len(existing_qa) >= 3:
            print(f"Skipping {qa_type} - already have {len(existing_qa)} entries (test mode limit: 3)")
            total_results[qa_type] = 0
            continue

        # Open file in append mode if resume, otherwise write mode
        mode = 'a' if args.resume and os.path.exists(output_file_path) else 'w'
        print(f"Opening output file {output_file_path} in {mode} mode")

        with open(output_file_path, mode, encoding='utf-8') as output_file:
            # Load data
            data = load_complex_paths_data(input_file)
            print(f"Loaded {len(data)} entries from {input_file}")

            # Filter out already processed entries if resume mode
            if args.resume and existing_qa:
                filtered_data = [entry for entry in data if entry.get('path_id') not in existing_qa]
                print(f"Filtered to {len(filtered_data)} entries (skipped {len(data) - len(filtered_data)} already processed)")
                data = filtered_data

            # In test mode, if we have some existing entries, only process until we reach 3 total
            if qa_config_with_nodes.get("test_mode", False) and len(existing_qa) > 0:
                remaining_needed = 3 - len(existing_qa)
                if remaining_needed <= 0:
                    print(f"Skipping {qa_type} - already have {len(existing_qa)} entries")
                    total_results[qa_type] = 0
                    continue
                else:
                    print(f"Need {remaining_needed} more entries for {qa_type} (current: {len(existing_qa)})")
                    # Limit data to remaining needed
                    data = data[:remaining_needed]

            # Generate QA based on type
            if qa_type == "parallel_diff_type":
                count = generate_parallel_diff_type_qa(data, forward_qa_mapping, qa_config_with_nodes, output_file)
            elif qa_type == "combo":
                count = generate_combo_qa(data, forward_qa_mapping, qa_config_with_nodes, output_file)
            elif qa_type == "parallel_same_entity":
                count = generate_parallel_same_entity_qa(data, forward_qa_mapping, qa_config_with_nodes, output_file)
            elif qa_type == "parallel_same_numerical":
                count = generate_parallel_same_numerical_qa(data, forward_qa_mapping, qa_config_with_nodes, output_file)
            else:
                print(f"Unknown QA type: {qa_type}")
                continue

            total_results[qa_type] = count
            print(f"Generated {count} entries for {qa_type}")

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for qa_type, count in total_results.items():
        print(f"{qa_type}: {count} entries generated")
    print(f"Total: {sum(total_results.values())} entries generated")


if __name__ == "__main__":
    main()