#!/usr/bin/env python3
"""
Script to validate complex paths uniqueness:
1. Check if all paths in a complex path combination involve different nodes
2. Check if each complex path combination is unique
"""

import json
import os
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


def load_edge_data(edge_ids: List[str], edges_file: str = "outputs/edges/all_edges.jsonl") -> Dict[str, Dict]:
    """Load edge data from edges file"""
    edge_data = {}
    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    edge = json.loads(line)
                    edge_id = edge.get('edge_id')
                    if edge_id in edge_ids:
                        edge_data[edge_id] = edge
    except FileNotFoundError:
        print(f"Warning: Edges file {edges_file} not found")
    return edge_data


def extract_nodes_from_path_segments(path_edges: List[List[str]], edges_data: Dict[str, Dict]) -> Set[str]:
    """Extract all unique node IDs from path segments"""
    nodes = set()
    for segment_edges in path_edges:
        for edge_id in segment_edges:
            if edge_id in edges_data:
                edge = edges_data[edge_id]
                nodes.add(edge.get('source_id'))
                nodes.add(edge.get('target_id'))
    return nodes


def check_node_overlap(path_edges: List[List[str]], edges_data: Dict[str, Dict]) -> bool:
    """Check if there's node overlap between path segments"""
    if len(path_edges) < 2:
        return False

    segment_nodes = []
    for segment_edges in path_edges:
        nodes = set()
        for edge_id in segment_edges:
            if edge_id in edges_data:
                edge = edges_data[edge_id]
                nodes.add(edge.get('source_id'))
                nodes.add(edge.get('target_id'))
        segment_nodes.append(nodes)

    # Check for overlap between any two segments
    for i in range(len(segment_nodes)):
        for j in range(i + 1, len(segment_nodes)):
            if segment_nodes[i] & segment_nodes[j]:  # Intersection exists
                return True

    return False


def get_path_combination_signature(path_edges: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
    """Create a signature for path combination to check uniqueness"""
    # Sort edges within each segment and segments themselves for consistent comparison
    normalized_segments = []
    for segment in path_edges:
        normalized_segments.append(tuple(sorted(segment)))

    # Sort segments by their first edge (or by length if empty)
    normalized_segments.sort()

    return tuple(normalized_segments)


def validate_complex_paths():
    """Main validation function"""
    complex_paths_dir = "outputs/complex_paths"

    if not os.path.exists(complex_paths_dir):
        print(f"Error: Directory {complex_paths_dir} not found")
        return

    # Get all JSONL files
    jsonl_files = [f for f in os.listdir(complex_paths_dir) if f.endswith('.jsonl')]

    if not jsonl_files:
        print(f"No JSONL files found in {complex_paths_dir}")
        return

    print(f"Found {len(jsonl_files)} files to validate")

    # Load all edge data once
    all_edge_ids = set()
    file_paths = {}

    for filename in jsonl_files:
        filepath = os.path.join(complex_paths_dir, filename)
        file_paths[filename] = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    path_data = json.loads(line)
                    file_paths[filename].append(path_data)

                    # Collect all edge IDs
                    path_edges = path_data.get('path_edges', [])
                    for segment in path_edges:
                        all_edge_ids.update(segment)

    print(f"Loading {len(all_edge_ids)} edges...")
    edges_data = load_edge_data(list(all_edge_ids))
    print(f"Loaded {len(edges_data)} edges")

    # Validation results
    validation_results = {
        'files_processed': 0,
        'total_complex_paths': 0,
        'paths_with_node_overlap': [],
        'duplicate_path_combinations': [],
        'path_combination_signatures': defaultdict(list)
    }

    # Process each file
    for filename, paths in file_paths.items():
        print(f"\nProcessing {filename}...")
        validation_results['files_processed'] += 1

        for i, path_data in enumerate(paths):
            validation_results['total_complex_paths'] += 1
            path_id = path_data.get('path_id', f'path_{i}')
            path_edges = path_data.get('path_edges', [])
            original_path_ids = path_data.get('original_path_ids', [])

            # Check 1: Node overlap within path segments
            has_node_overlap = check_node_overlap(path_edges, edges_data)
            if has_node_overlap:
                validation_results['paths_with_node_overlap'].append({
                    'file': filename,
                    'path_id': path_id,
                    'original_path_ids': original_path_ids,
                    'path_edges': path_edges
                })

            # Check 2: Unique path combinations
            signature = get_path_combination_signature(path_edges)
            validation_results['path_combination_signatures'][signature].append({
                'file': filename,
                'path_id': path_id,
                'original_path_ids': original_path_ids,
                'path_data': path_data
            })

    # Find duplicate combinations
    for signature, occurrences in validation_results['path_combination_signatures'].items():
        if len(occurrences) > 1:
            validation_results['duplicate_path_combinations'].append({
                'signature': signature,
                'occurrences': occurrences
            })

    # Print results
    print(f"\n{'='*50}")
    print("VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Files processed: {validation_results['files_processed']}")
    print(f"Total complex paths: {validation_results['total_complex_paths']}")

    print(f"\n{'='*30}")
    print("NODE OVERLAP ISSUES")
    print(f"{'='*30}")

    if validation_results['paths_with_node_overlap']:
        print(f"⚠️  Found {len(validation_results['paths_with_node_overlap'])} paths with node overlap:")
        for issue in validation_results['paths_with_node_overlap']:
            print(f"  📁 File: {issue['file']}")
            print(f"     Path ID: {issue['path_id']}")
            print(f"     Original paths: {issue['original_path_ids']}")
            print(f"     Path edges: {issue['path_edges']}")
            print()
    else:
        print("✅ No paths with node overlap found!")

    print(f"\n{'='*30}")
    print("DUPLICATE PATH COMBINATIONS")
    print(f"{'='*30}")

    if validation_results['duplicate_path_combinations']:
        print(f"⚠️  Found {len(validation_results['duplicate_path_combinations'])} duplicate path combinations:")
        for dup in validation_results['duplicate_path_combinations']:
            print(f"  🔗 Signature: {dup['signature']}")
            print(f"     Occurrences ({len(dup['occurrences'])}):")
            for occ in dup['occurrences']:
                print(f"       - {occ['file']}: {occ['path_id']} (paths: {occ['original_path_ids']})")
            print()
    else:
        print("✅ All path combinations are unique!")

    # Summary
    total_issues = len(validation_results['paths_with_node_overlap']) + len(validation_results['duplicate_path_combinations'])
    if total_issues == 0:
        print(f"\n🎉 All validations passed! No issues found.")
    else:
        print(f"\n⚠️  Found {total_issues} total issues that need attention.")


if __name__ == "__main__":
    validate_complex_paths()