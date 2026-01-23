# Mini-Wiki Data Generation Pipeline

## Overview

This pipeline generates a synthetic knowledge graph with Wikipedia-style articles and corresponding QA pairs. The process starts from name generation, builds nodes and edges, constructs reasoning paths, and finally generates simple and complex QA pairs.

## Pipeline Architecture

### Core Components

1. **Unified Configuration** (`scripts/unified_builder.py`)
   - Centralized configuration management
   - Controls quantities, file paths, and QA generation settings

2. **Main Orchestrator** (`scripts/run_all_steps.py`)
   - Command-line interface for the entire pipeline
   - Supports step-by-step or full execution
   - Auto-detects existing files for incremental runs

3. **Modular Scripts**
   - Each step is an independent script
   - Can be executed individually or combined

## Execution Flow

```
1. Generate Names (generate_names.py)
   |
   v
2. Build Nodes (build_nodes.py)
   |
   v
3. Build Knowledge Graph (build_kg.py)
   |
   v
4. Build Simple Paths (build_paths.py)
   |
   v
5. Build Complex Paths (build_complex_paths.py)
   |
   v
6a. Generate Simple QA (generate_simple_qa.py)
6b. Generate Complex QA (generate_complex_qa.py)
```

## Step Details

### Step 1: Generate Names (`scripts/generate_names.py`)

Generates diverse names for different entity types:
- Countries (default: 200)
- Cities (default: 400)
- Persons (default: 2000)
- Universities (default: 200)
- Companies (default: 200)
- Languages (default: 50)

**Output:** `outputs/names/names_{type}.json`

### Step 2: Build Nodes (`scripts/build_nodes.py`)

Creates knowledge graph nodes with attributes:
- **Country**: population, GDP, area, etc.
- **City**: population, area, located country, etc.
- **University**: student count, professor count, ranking, etc.
- **Company**: employee count, product count, etc.
- **Person**: birth year, gender, education, relationships, etc.

**Output:** `outputs/nodes/nodes_{type}.jsonl`

### Step 3: Build Knowledge Graph (`scripts/build_kg.py`)

Constructs edges between nodes based on attributes:
- Forward and reverse relationships
- Mapping types: 1-1, 1-n, n-1
- `need_access` flag for document retrieval requirement

**Output:**
- `outputs/edges/forward_edges.jsonl`
- `outputs/edges/reverse_edges.jsonl`
- `outputs/edges/all_edges.jsonl`
- `outputs/edges/graph_summary.json`

### Step 4: Build Simple Paths (`scripts/build_paths.py`)

Finds reasoning paths in the knowledge graph:
- 1-hop to 6-hop paths
- Classified by `need_access` requirement
- Ensures path uniqueness

**Output:**
- `outputs/paths/forward_paths.jsonl`
- `outputs/paths/forward_paths_summary.json`

### Step 5: Build Complex Paths (`scripts/build_complex_paths.py`)

Constructs complex reasoning paths:
- **Parallel Numerical**: Compare two numerical values
- **Parallel Entity**: Compare same-type entities
- **Combo**: Long paths (7-12 hops)

**Output:**
- `outputs/complex_paths/parallel_same_type_numerical_qa_paths.jsonl`
- `outputs/complex_paths/parallel_diff_type_qa_paths.jsonl`
- `outputs/complex_paths/combo_qa_paths.jsonl`

### Step 6a: Generate Simple QA (`scripts/generate_simple_qa.py`)

Generates QA pairs from simple paths using multiple templates:
- 1-3 hop: 3 template variants each
- 4-6 hop: 2 template variants each

**Output:** `outputs/qa/{n}-hop.jsonl`, `outputs/qa/forward_qa.jsonl`

### Step 6b: Generate Complex QA (`scripts/generate_complex_qa.py`)

Generates QA pairs from complex paths:
- Combo questions
- Parallel numerical comparison
- Parallel entity comparison

**Output:** `outputs/qa/{complex_type}_qa.jsonl`

## Usage

### Quick Start

```bash
# Run complete pipeline
python scripts/run_all_steps.py --steps all

# Run specific steps
python scripts/run_all_steps.py --steps names nodes kg

# Generate QA only (requires existing paths)
python scripts/run_all_steps.py --steps simple_qa complex_qa

# Show current configuration
python scripts/run_all_steps.py --show-config

# Force regenerate names
python scripts/run_all_steps.py --steps names --force-names
```

### Run Individual Scripts

```bash
python scripts/generate_names.py --type country city person
python scripts/build_nodes.py
python scripts/build_kg.py
python scripts/build_paths.py
python scripts/build_complex_paths.py
python scripts/generate_simple_qa.py
python scripts/generate_complex_qa.py
```

## Configuration

Edit `scripts/unified_builder.py` to modify:

```python
UNIFIED_CONFIG = {
    "name_generation": {
        "country": 200,
        "city": 400,
        "person": 2000,
        ...
    },
    "path_generation": {
        "forward_paths_need_access": {
            "1-hop": 6000,
            "2-hop": 5000,
            ...
        }
    },
    "qa_generation": {
        "generate_path": True,
        "generate_qa": False
    }
}
```

### Execution Modes

| Mode | Config | Steps Executed |
|------|--------|----------------|
| Paths Only | `generate_path=True, generate_qa=False` | names → nodes → kg → paths → complex_paths |
| QA Only | `generate_path=False, generate_qa=True` | simple_qa → complex_qa |
| Full | `generate_path=True, generate_qa=True` | All steps |

## Project Structure

```
mini-wiki/
├── scripts/
│   ├── run_all_steps.py          # Main orchestrator
│   ├── unified_builder.py        # Configuration
│   ├── generate_names.py         # Step 1
│   ├── build_nodes.py            # Step 2
│   ├── build_kg.py               # Step 3
│   ├── build_paths.py            # Step 4
│   ├── build_complex_paths.py    # Step 5
│   ├── generate_simple_qa.py     # Step 6a
│   ├── generate_complex_qa.py    # Step 6b
│   ├── generate_wiki.py          # Wiki article generation
│   └── generate_urls_for_wiki.py # URL generation
├── outputs/
│   ├── names/                    # Generated names
│   ├── nodes/                    # Node data
│   ├── edges/                    # Edge relationships
│   ├── paths/                    # Simple paths
│   ├── complex_paths/            # Complex paths
│   ├── qa/                       # QA pairs
│   └── wiki/                     # Wiki articles
├── prompts/                      # LLM prompt templates
├── utils/                        # Utility functions
├── website_template/             # HTML templates for wiki
└── schema_config.json            # Node schema configuration
```

## Validation Scripts

Optional validation tools (not part of main pipeline):
- `scripts/validate_path_uniqueness.py` - Check path uniqueness
- `scripts/validate_complex_paths.py` - Validate complex paths
- `scripts/analyze_template_usage.py` - Analyze template distribution
- `scripts/annotate_path_search_strategy.py` - Annotate search strategies

## Requirements

- Python 3.7+
- LLM API access (configure in `utils/llm_utils_ds.py`)
- Environment variable: `DEEPSEEK_API_KEY`

## Notes

1. **Incremental Runs**: The pipeline auto-detects existing files and skips completed steps
2. **Reproducibility**: Use fixed random seeds in configuration
3. **API Rate Limits**: Adjust `NUM_WORKERS` and sleep intervals as needed
4. **Memory**: For large-scale generation, monitor memory usage
