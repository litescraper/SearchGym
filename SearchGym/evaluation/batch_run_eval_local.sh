#!/bin/bash

# Batch evaluation script for multiple models using run_eval_local.sh configuration
# This script processes multiple models using searchgym agent type and searchgym prompt type

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Set environment variables for 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Common configuration
DATA_DIR=../../SearchGym-test-data
SPLIT=1
MAX_GEN_TOKENS=4096
DATA_NAMES=Bamboogle
AGENT_TYPE=SearchGym
PROMPT_TYPE=SearchGym
SEARCH_CLIENT_TYPE=async-search-access
temperature=0.6
top_p=0.95
top_k=-1

# Model list to process - ADD YOUR MODELS HERE
MODEL_PATHS=(
    "/path/to/your/model/checkpoint"  # Add your model checkpoint paths here
)

# Output directory for logs
OUTPUT_DIR="outputs"
mkdir -p $OUTPUT_DIR

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/batch_eval_local_${TIMESTAMP}.log"

echo "Starting batch evaluation for searchgym models..."
echo "Using 8 GPUs: $CUDA_VISIBLE_DEVICES"
echo "Total models to process: ${#MODEL_PATHS[@]}"
echo "Log file: $LOG_FILE"
echo "=================================================="

# Function to evaluate a single model
evaluate_model() {
    local model_path=$1
    local model_name=$(basename $model_path)
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    echo "" | tee -a $LOG_FILE
    echo "Processing model: $model_path" | tee -a $LOG_FILE
    echo "Model name: $model_name" | tee -a $LOG_FILE
    echo "Start time: $(date)" | tee -a $LOG_FILE
    echo "--------------------------------------------------" | tee -a $LOG_FILE

    # Check if model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path" | tee -a $LOG_FILE
        echo "Skipping..." | tee -a $LOG_FILE
        return 1
    fi

    # Create model-specific output directory
    local model_output_dir="${OUTPUT_DIR}/${model_name}"
    mkdir -p $model_output_dir

    # Run evaluation
    echo "Running evaluation for $model_name..." | tee -a $LOG_FILE
    echo "MODEL PATH: $model_path" | tee -a $LOG_FILE
    echo "Agent Type: $AGENT_TYPE" | tee -a $LOG_FILE
    echo "Prompt Type: $PROMPT_TYPE" | tee -a $LOG_FILE
    echo "Temperature: $temperature" | tee -a $LOG_FILE

    local eval_log_file="${model_output_dir}/eval_local_${timestamp}.log"

    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="../..:$PYTHONPATH" \
    python3 search_eval_async.py \
        --data_names ${DATA_NAMES} \
        --model_name_or_path ${model_path}  \
        --output_dir ${model_path} \
        --prompt_type $PROMPT_TYPE \
        --agent-type ${AGENT_TYPE} \
        --data_dir ${DATA_DIR} \
        --split test \
        --search-client-type ${SEARCH_CLIENT_TYPE} \
        --max-tokens-per-call ${MAX_GEN_TOKENS} \
        --tensor_parallel_size 4 \
        --n_sampling 1 \
        --temperature ${temperature} \
        --top_p $top_p \
        --top_k $top_k \
        --start 0 \
        --end -1 \
        --seed 1 \
        --parallel-mode seed \
        --llm_as_judge \
        --pass-at-k 1 \
        --config-path eval_config.yaml \
        2>&1 | tee -a $eval_log_file

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: $model_name" | tee -a $LOG_FILE
        echo "Results saved in: $model_output_dir" | tee -a $LOG_FILE
    else
        echo "✗ Evaluation failed for: $model_name (Exit code: $exit_code)" | tee -a $LOG_FILE
    fi

    echo "End time: $(date)" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE

    return $exit_code
}

# Process all models
total_models=${#MODEL_PATHS[@]}
echo "Total models to process: $total_models" | tee -a $LOG_FILE

if [ $total_models -eq 0 ]; then
    echo "Warning: No models specified in MODEL_PATHS array!" | tee -a $LOG_FILE
    echo "Please edit the script and add your model paths." | tee -a $LOG_FILE
    exit 1
fi

successful_evals=0
failed_evals=0

for i in "${!MODEL_PATHS[@]}"; do
    current_model=${MODEL_PATHS[$i]}
    echo "" | tee -a $LOG_FILE
    echo "Processing model $((i+1))/$total_models" | tee -a $LOG_FILE

    if evaluate_model "$current_model"; then
        ((successful_evals++))
    else
        ((failed_evals++))
    fi

    # Brief pause between models
    if [ $((i+1)) -lt $total_models ]; then
        echo "Waiting 30 seconds before processing next model..." | tee -a $LOG_FILE
        sleep 30
    fi
done

# Final summary
echo "" | tee -a $LOG_FILE
echo "Batch evaluation completed!" | tee -a $LOG_FILE
echo "Total models processed: $total_models" | tee -a $LOG_FILE
echo "Successful evaluations: $successful_evals" | tee -a $LOG_FILE
echo "Failed evaluations: $failed_evals" | tee -a $LOG_FILE
echo "All results are saved in: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

# Summary of processed models
echo "" | tee -a $LOG_FILE
echo "Summary of processed models:" | tee -a $LOG_FILE
for i in "${!MODEL_PATHS[@]}"; do
    model_path=${MODEL_PATHS[$i]}
    model_name=$(basename $model_path)
    model_output_dir="${OUTPUT_DIR}/${model_name}"

    if [ -d "$model_output_dir" ]; then
        log_files=$(ls $model_output_dir/eval_local_*.log 2>/dev/null | wc -l)
        echo "  $model_name: $log_files evaluation log(s) found" | tee -a $LOG_FILE
    else
        echo "  $model_name: No results (skipped or failed)" | tee -a $LOG_FILE
    fi
done

echo "" | tee -a $LOG_FILE
echo "Batch eval local script finished at $(date)" | tee -a $LOG_FILE