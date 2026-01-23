#!/bin/bash

# Set your project root path
SEARCHGYM_ROOT="/path/to/SearchGym"

cd ${SEARCHGYM_ROOT}/AReaL &&
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 &&
export WANDB_API_KEY=your-wandb-api-key &&
export PYTHONPATH=${SEARCHGYM_ROOT}/SearchGym:${SEARCHGYM_ROOT}/AReaL &&
python -m areal.launcher.local ../SearchGym/SearchGym/train/train_main.py \
  --config ../SearchGym/SearchGym/configs/SearchGym_stage2.yaml \
  +experiment_name=agent_rl \
  +trial_name=code_test2