#!/usr/bin/env python3
"""
Script to analyze LLM judge results from agent_eval_4096 directories
across multiple models and datasets.
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict

def read_jsonl_file(file_path):
    """Read and parse JSONL file"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def extract_judge_score(entry):
    """Extract score from LLM judge entry"""
    try:
        if 'llm_as_judge' in entry and entry['llm_as_judge']:
            judge_info = entry['llm_as_judge']
            if isinstance(judge_info, dict) and 'judgement' in judge_info:
                judgement = judge_info['judgement'].lower()
                return 1 if judgement == 'correct' else 0
        return 0
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 0

def analyze_checkpoint(checkpoint_path):
    """Analyze a single checkpoint directory"""
    agent_eval_path = os.path.join(checkpoint_path, 'agent_eval_4096')
    if not os.path.exists(agent_eval_path):
        return None

    # Read aggregate results first - look for any aggregate_results*.json file
    aggregate_files = glob.glob(os.path.join(agent_eval_path, 'aggregate_results*.json'))
    aggregate_results = {}
    if aggregate_files:
        # Use the first aggregate_results file found
        aggregate_file = aggregate_files[0]
        try:
            with open(aggregate_file, 'r') as f:
                aggregate_results = json.load(f)
        except Exception as e:
            print(f"Error reading aggregate results {aggregate_file}: {e}")

    # Analyze each dataset
    datasets = ['NQ_rand1000', 'TriviaQA_rand1000', 'PopQA_rand1000', 'HotpotQA_rand1000',
                '2WikiMultihopQA_rand1000', 'Musique_rand1000', 'Bamboogle']

    results = {
        'checkpoint_path': checkpoint_path,
        'aggregate_results': aggregate_results,
        'total_score': 0,
        'total_questions': 0
    }

    for dataset in datasets:
        dataset_path = os.path.join(agent_eval_path, dataset)
        if not os.path.exists(dataset_path):
            continue

        # Look for the LLM judge JSONL file
        judge_files = glob.glob(os.path.join(dataset_path, '*llm_as_judge*.jsonl'))
        if not judge_files:
            print(f"No judge file found for {dataset} in {checkpoint_path}")
            continue

        judge_file = judge_files[0]  # Take the first match
        entries = read_jsonl_file(judge_file)

        scores = []
        for entry in entries:
            score = extract_judge_score(entry)
            scores.append(score)

        if scores:
            dataset_total = sum(scores)
            dataset_count = len(scores)
            dataset_avg = dataset_total / dataset_count

            # Add LLM judge scores directly to aggregate results
            if dataset not in aggregate_results:
                aggregate_results[dataset] = {}

            aggregate_results[dataset]['LLM_Judge_Accuracy'] = dataset_avg
            aggregate_results[dataset]['LLM_Judge_Correct'] = dataset_total
            aggregate_results[dataset]['LLM_Judge_Total'] = dataset_count

            # Calculate improvement over original EM score
            if 'EM' in aggregate_results[dataset]:
                original_em = aggregate_results[dataset]['EM']
                improvement = dataset_avg - original_em
                aggregate_results[dataset]['EM_Improvement'] = improvement

            results['total_score'] += dataset_total
            results['total_questions'] += dataset_count

    # Calculate overall average
    if results['total_questions'] > 0:
        results['overall_average'] = results['total_score'] / results['total_questions']
    else:
        results['overall_average'] = 0.0

    return results

def main():
    checkpoint_paths = [
        "/path/to/your/model/checkpoint"  # Add your checkpoint paths here
    ]
    all_results = {}

    print("Analyzing all checkpoints...")
    for checkpoint_path in checkpoint_paths:
        print(f"\nAnalyzing: {checkpoint_path}")

        # Extract model name from path for cleaner display, including step info
        path_parts = checkpoint_path.split('/')
        # For paths like: .../agent_rl/7B_mix_access_.../default/epoch1epochstep141globalstep419
        # Extract base model (usually -3) and step info (usually -1)
        if len(path_parts) >= 3:
            # Check if the last part looks like a step info (contains 'epoch' and 'step')
            last_part = path_parts[-1]
            if 'epoch' in last_part and 'step' in last_part:
                # This is step info, get base model from -3 position
                base_model = path_parts[-3]  # e.g., "7B_mix_access_mix_access_ablation_hop_1-9"
                step_info = path_parts[-1]   # e.g., "epoch1epochstep141globalstep419"
                model_name = f"{base_model}_{step_info}"
            else:
                # For direct model paths without step info
                model_name = last_part
        else:
            model_name = checkpoint_path.split('/')[-1]  # Fallback
        print(f"Model: {model_name}")

        results = analyze_checkpoint(checkpoint_path)
        if results:
            all_results[model_name] = results

            # Print summary for this model
            print(f"  Total Score: {results['total_score']}/{results['total_questions']}")
            print(f"  Overall Average: {results['overall_average']:.3f}")
        else:
            print(f"  No results found for {checkpoint_path}")

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    # Table header
    print(f"{'Model':<40} {'Total':<10} {'Questions':<10} {'Average':<10} {'Datasets'}")
    print("-" * 100)

    for model_name, results in all_results.items():
        dataset_count = len([d for d in results['aggregate_results'] if 'LLM_Judge_Accuracy' in results['aggregate_results'][d]])
        print(f"{model_name:<40} {results['total_score']:<10} {results['total_questions']:<10} {results['overall_average']:<10.3f} {dataset_count}")

    # Generate EM scores table
    print("\n" + "="*120)
    print("ORIGINAL EM SCORES ACROSS ALL DATASETS")
    print("="*120)

    datasets = ['NQ_rand1000', 'TriviaQA_rand1000', 'PopQA_rand1000', 'HotpotQA_rand1000',
                '2WikiMultihopQA_rand1000', 'Musique_rand1000', 'Bamboogle']

    # Shorten dataset names for table display
    dataset_short_names = ['NQ', 'TriviaQA', 'PopQA', 'HotpotQA', '2WikiQA', 'Musique', 'Bamboogle']

    # Table header
    print(f"{'Model':<45}", end="")
    for dataset_name in dataset_short_names:
        print(f"{dataset_name:<12}", end="")
    print("Average")
    print("-" * 140)

    # Table rows for EM scores
    for model_name, results in all_results.items():
        print(f"{model_name:<45}", end="")

        em_scores = []
        for dataset in datasets:
            if dataset in results['aggregate_results'] and 'EM' in results['aggregate_results'][dataset]:
                em_score = results['aggregate_results'][dataset]['EM']
                em_scores.append(em_score)
                print(f"{em_score:<12.3f}", end="")
            else:
                print(f"{'N/A':<12}", end="")

        # Calculate average
        if em_scores:
            avg_em = sum(em_scores) / len(em_scores)
            print(f"{avg_em:<12.3f}")
        else:
            print(f"{'N/A':<12}")

    # Generate LLM Judge scores table
    print("\n" + "="*120)
    print("LLM JUDGE SCORES ACROSS ALL DATASETS")
    print("="*120)

    # Table header
    print(f"{'Model':<45}", end="")
    for dataset_name in dataset_short_names:
        print(f"{dataset_name:<12}", end="")
    print("Average")
    print("-" * 140)

    # Table rows for LLM Judge scores
    for model_name, results in all_results.items():
        print(f"{model_name:<45}", end="")

        accuracies = []
        for dataset in datasets:
            if dataset in results['aggregate_results'] and 'LLM_Judge_Accuracy' in results['aggregate_results'][dataset]:
                accuracy = results['aggregate_results'][dataset]['LLM_Judge_Accuracy']
                accuracies.append(accuracy)
                print(f"{accuracy:<12.3f}", end="")
            else:
                print(f"{'N/A':<12}", end="")

        # Calculate average
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"{avg_accuracy:<12.3f}")
        else:
            print(f"{'N/A':<12}")

    # Generate EM Improvement table
    print("\n" + "="*120)
    print("EM IMPROVEMENT (LLM Judge - Original EM)")
    print("="*120)

    # Table header
    print(f"{'Model':<45}", end="")
    for dataset_name in dataset_short_names:
        print(f"{dataset_name:<12}", end="")
    print("Average")
    print("-" * 140)

    # Table rows for EM improvements
    for model_name, results in all_results.items():
        print(f"{model_name:<45}", end="")

        improvements = []
        for dataset in datasets:
            if (dataset in results['aggregate_results'] and
                'EM_Improvement' in results['aggregate_results'][dataset]):
                improvement = results['aggregate_results'][dataset]['EM_Improvement']
                improvements.append(improvement)
                # Show improvement with + sign for positive values
                improvement_str = f"{improvement:+.3f}"
                print(f"{improvement_str:<12}", end="")
            else:
                print(f"{'N/A':<12}", end="")

        # Calculate average improvement
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            avg_improvement_str = f"{avg_improvement:+.3f}"
            print(f"{avg_improvement_str:<12}")
        else:
            print(f"{'N/A':<12}")

    # Generate summary with total scores
    print("\n" + "="*120)
    print("TOTAL SCORES SUMMARY")
    print("="*120)

    print(f"{'Model':<45} {'Total Score':<15} {'Total Questions':<20} {'Overall Average':<15}")
    print("-" * 100)

    for model_name, results in all_results.items():
        dataset_count = len([d for d in results['aggregate_results'] if 'LLM_Judge_Accuracy' in results['aggregate_results'][d]])
        print(f"{model_name:<45} {results['total_score']:<15} {results['total_questions']:<20} {results['overall_average']:<15.3f}")

    # Save results to JSON file
    output_file = 'llm_judge_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()