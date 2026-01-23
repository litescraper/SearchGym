# SearchGym

## Pre-trained Models

We provide pre-trained SearchGym models on HuggingFace: [SearchGym Collection](https://huggingface.co/collections/hkuzxc/searchgym)

| Model | Size | Link |
|-------|------|------|
| SearchGym_Qwen_2.5_3B_Base | 3B | [hkuzxc/SearchGym_Qwen_2.5_3B_Base](https://huggingface.co/hkuzxc/SearchGym_Qwen_2.5_3B_Base) |
| SearchGym_Qwen_2.5_3B_Instruct | 3B | [hkuzxc/SearchGym_Qwen_2.5_3B_Instruct](https://huggingface.co/hkuzxc/SearchGym_Qwen_2.5_3B_Instruct) |
| SearchGym_Qwen_2.5_7B_Base | 7B | [hkuzxc/SearchGym_Qwen_2.5_7B_Base](https://huggingface.co/hkuzxc/SearchGym_Qwen_2.5_7B_Base) |
| SearchGym_Qwen_2.5_7B_Instruct | 7B | [hkuzxc/SearchGym_Qwen_2.5_7B_Instruct](https://huggingface.co/hkuzxc/SearchGym_Qwen_2.5_7B_Instruct) |
| SearchGym_Qwen_3_4B | 4B | [hkuzxc/SearchGym_Qwen_3_4B](https://huggingface.co/hkuzxc/SearchGym_Qwen_3_4B) |
| SearchGym_Qwen_3_8B | 8B | [hkuzxc/SearchGym_Qwen_3_8B](https://huggingface.co/hkuzxc/SearchGym_Qwen_3_8B) |
| SearchGym_Llama_3.2_3B_Instruct | 3B | [hkuzxc/SearchGym_Llama_3.2_3B_Instruct](https://huggingface.co/hkuzxc/SearchGym_Llama_3.2_3B_Instruct) |

## 1. Training Environment Setup

```bash
# Create conda environment
conda create -n SearchGym python=3.12
conda activate SearchGym

# Install dependencies
cd AReal
bash examples/env/setup-pip-deps.sh

# Validate installation
python examples/env/validate_installation.py
```

## 2. Mini-Wiki Data Generation (Optional)

Generate synthetic knowledge graph and QA pairs for training/evaluation.

### Prerequisites

```bash
# Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Or create a .env file in mini-wiki/
echo 'DEEPSEEK_API_KEY=your-api-key-here' > mini-wiki/.env
```

### Generate Data

```bash
cd mini-wiki

# Run the complete pipeline
python scripts/run_all_steps.py --steps all

# Or run specific steps
python scripts/run_all_steps.py --steps names nodes kg paths

# Show configuration
python scripts/run_all_steps.py --show-config
```

See [mini-wiki/README_DATA_GENERATION.md](mini-wiki/README_DATA_GENERATION.md) for detailed documentation.

## 3. Search Environment Setup (Meilisearch)

```bash
# Install Meilisearch
curl -L https://install.meilisearch.com | sh

# Start Meilisearch server
mkdir -p logs && nohup ./meilisearch --master-key="aSampleMasterKey" > logs/meilisearch.log 2>&1 &

# Index the wiki data
curl \
  -X POST 'http://127.0.0.1:7700/indexes/wiki/documents?primaryKey=id' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer aSampleMasterKey' \
  --data-binary @mini-wiki/outputs/wiki/wiki_with_urls.json
```

## 4. Download Training & Evaluation Data

Download the dataset from HuggingFace:

```bash

git clone https://huggingface.co/datasets/hkuzxc/SearchGym-test-data

# The dataset should be placed in the project root directory
# Final structure: SearchGym/SearchGym-test-data/
```

The dataset [hkuzxc/SearchGym-test-data](https://huggingface.co/datasets/hkuzxc/SearchGym-test-data) contains:
- `mini_wiki_train/` - Training data (stage1 & stage2)
- `mini_wiki_test/` - Test data for Mini-Wiki
- `Bamboogle/`, `HotpotQA_rand1000/`, `Musique_rand1000/`, etc. - Evaluation benchmarks

## 5. Configuration (Before Training)

Before running training or evaluation, you need to configure the following files with your own paths:

### Training Configuration

1. **`SearchGym/run_SearchGym_stage1.sh`** and **`SearchGym/run_SearchGym_stage2.sh`**:
   ```bash
   SEARCHGYM_ROOT="/path/to/SearchGym"  # Your project root path
   WANDB_API_KEY=your-wandb-api-key     # Your Weights & Biases API key
   ```

2. **`SearchGym/SearchGym/configs/SearchGym_stage1.yaml`** (and `stage2.yaml`):
   ```yaml
   cluster:
     fileroot: /path/to/experiments        # Experiment output directory
     name_resolve:
       nfs_record_root: /path/to/name_resolve
   
   actor:
     path: /path/to/your/model             # Base model path (e.g., Qwen2.5-3B)
   
   # Data paths are pre-configured as relative paths:
   # train_dataset.path: ../SearchGym-test-data/mini_wiki_train/stage1/stage1_train.jsonl
   # valid_dataset.path: ../SearchGym-test-data/mini_wiki_train/stage1/stage1_test.jsonl
   
   launcher:
     trainer_env_vars: PYTHONPATH=/path/to/SearchGym/SearchGym:/path/to/SearchGym/AReaL,WANDB_API_KEY=your-wandb-api-key
   ```

### Evaluation Configuration

3. **`SearchGym/evaluation/eval_config.yaml`**:
   ```yaml
   api_keys:
     serper_api_key: "your-serper-api-key"   # Get from https://serper.dev/
     jina_api_key: "your-jina-api-key"       # Get from https://jina.ai/
   
   settings:
     local_server:
       address: "your-server-ip"
       port: "8000"
   ```

4. **`SearchGym/scripts/build_index.sh`** and **`launch_local_server.sh`**:
   ```bash
   WIKI2018_WORK_DIR=/path/to/your/retrieval-data
   retriever_model=/path/to/your/e5-model
   ```

### Optional: Cache Directory

Set the `AREAL_CACHE_DIR` environment variable to customize the cache location:
```bash
export AREAL_CACHE_DIR=/path/to/your/cache
```

## 6. Run Training

```bash
cd SearchGym
bash run_SearchGym_stage1.sh
```

## 7. Evaluation Environment Setup

### Local RAG Server (for Bamboogle, 2WikiMultihopQA, HotpotQA, Musique, NQ, PopQA, TriviaQA)

```bash
# Create separate environment
conda create -n retriever python=3.10
conda activate retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install transformers datasets pyserini

# Install GPU-accelerated FAISS
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Install API server dependencies
pip install uvicorn fastapi
```

### Download Required Files

1. Download [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) model
2. Download from [ASearcher-Local-Knowledge](https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/tree/main):
   - E5 retriever index file
   - Corpus file
   - Webpage file

### Build Index and Launch Server

```bash
# Build the index
bash scripts/build_index.sh

# Launch local RAG server
export PORT=8000
export RAG_SERVER_ADDR_DIR="/path/to/SearchGym/SearchGym/server_address"
nohup bash scripts/launch_local_server.sh $PORT $RAG_SERVER_ADDR_DIR > retrieval_launch.log 2>&1 &
```

### Configure Evaluation

Edit `SearchGym/evaluation/eval_config.yaml`:

```yaml
local_server:
  address: "your-server-ip"
  port: "8000"
```

## Acknowledgments

This project is built upon the outstanding work of:

- **[AReaL](https://github.com/inclusionAI/AReaL)** - A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning and Agents, developed by the AReaL Team at Ant Group and Tsinghua IIIS.
- **[ASearcher](https://github.com/inclusionAI/ASearcher)** - An Open-Source Large-Scale Reinforcement Learning Project for Search Agents.

We are deeply grateful to the authors and contributors of these projects for their pioneering work in asynchronous RL training and search agent development.

## Citation

If you find this project useful, please consider citing the following papers:

```bibtex
@misc{fu2025areal,
      title={AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning},
      author={Wei Fu and Jiaxuan Gao and Xujie Shen and Chen Zhu and Zhiyu Mei and Chuyi He and Shusheng Xu and Guo Wei and Jun Mei and Jiashu Wang and Tongkai Yang and Binhang Yuan and Yi Wu},
      year={2025},
      eprint={2505.24298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24298},
}

@misc{gao2025turnsunlockinglonghorizonagentic,
      title={Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL}, 
      author={Jiaxuan Gao and Wei Fu and Minyang Xie and Shusheng Xu and Chuyi He and Zhiyu Mei and Banghua Zhu and Yi Wu},
      year={2025},
      eprint={2508.07976},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07976}, 
}

@misc{zhang2026searchgymbootstrappingrealworldsearch,
      title={SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation}, 
      author={Xichen Zhang and Ziyi He and Yinghao Zhu and Sitong Wu and Shaozuo Yu and Meng Chu and Wenhu Zhang and Haoru Tan and Jiaya Jia},
      year={2026},
      eprint={2601.14615},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.14615}, 
}
```
