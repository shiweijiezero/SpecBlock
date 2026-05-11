#!/bin/bash
#
# Data generation with two modes:
#   1. offline: Direct Engine inference (faster, no HTTP overhead)
#   2. api: API server mode (original, for compatibility)
#
# Usage:
#   bash scripts/run_offline_generate.sh \
#       --model llama \
#       --mode offline \
#       --gpus 0,1,2,3 \
#       --input ./cache/dataset/train_dataset.jsonl \
#       --output ./cache/offline-generated/llama-3.1-8b-instruct
#

set -e

# ============== Configuration ==============
# Default values
MODEL_TYPE="llama"        # llama, vicuna, qwen, deepseek
MODE="offline"            # offline or api
GPU_LIST="0,1,2"
INPUT_FILE_ARG=""
OUTPUT_DIR_ARG=""
MAX_RETRIES=3
BATCH_SIZE=8192
MAX_CONCURRENCY=256
MAX_TOKENS=2048            # per-turn max_new_tokens (training truncates prompt+response to 1024, but we generate longer to preserve complete responses)

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE_ARG="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Model paths
declare -A MODEL_PATHS=(
    ["llama"]="meta-llama/Llama-3.1-8B-Instruct"
    ["vicuna"]="/path/to/hf_cache/vicuna-13b-v1.3"
    ["qwen"]="Qwen/Qwen2.5-7B-Instruct"
    ["qwen3"]="Qwen/Qwen3-8B"
    ["deepseek"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Input data files (ALL use full dataset since original had SYSTEM_PROMPT="")
declare -A INPUT_FILES=(
    ["llama"]="cache/dataset/train_dataset.jsonl"
    ["vicuna"]="cache/dataset/train_dataset.jsonl"
    ["qwen"]="cache/dataset/train_dataset.jsonl"
    ["qwen3"]="cache/dataset/train_dataset.jsonl"
    ["deepseek"]="cache/dataset/train_dataset.jsonl"
)

# Output directories
declare -A OUTPUT_DIRS=(
    ["llama"]="cache/offline-generated/llama-3.1-8b-instruct"
    ["vicuna"]="cache/offline-generated/vicuna-13b-v1.3"
    ["qwen"]="cache/offline-generated/qwen2.5-7b-instruct"
    ["qwen3"]="cache/offline-generated/qwen3-8b-instruct"
    ["deepseek"]="cache/offline-generated/deepseek-r1-distill-llama-8b"
)

# System prompts for each model
# Llama3: MUST pass safety prompt (tokenizer only auto-adds date info, not safety prompt)
# Qwen: Empty - tokenizer auto-adds "You are Qwen, created by Alibaba Cloud..."
# Qwen3: Empty - tokenizer auto-adds, and --enable-thinking controls thinking mode
# Vicuna: Empty - tokenizer has built-in conversation format
# DeepSeek-R1-Distill: Empty - no system prompt expected
declare -A SYSTEM_PROMPTS=(
    ["llama"]="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    ["vicuna"]=""
    ["qwen"]=""
    ["qwen3"]=""
    ["deepseek"]=""
)

# Qwen3 thinking mode: empty = instruct (default, faster, shorter output),
# non-empty (e.g. "--enable-thinking") = pass flag to python for thinking chain
declare -A EXTRA_FLAGS=(
    ["llama"]=""
    ["vicuna"]=""
    ["qwen"]=""
    ["qwen3"]=""
    ["deepseek"]=""
)

# ============== Setup ==============
MODEL_PATH="${MODEL_PATHS[$MODEL_TYPE]}"
SYSTEM_PROMPT="${SYSTEM_PROMPTS[$MODEL_TYPE]}"
EXTRA_FLAG="${EXTRA_FLAGS[$MODEL_TYPE]}"

# Use command line args if provided, otherwise use defaults
if [ -n "$INPUT_FILE_ARG" ]; then
    INPUT_FILE="$INPUT_FILE_ARG"
else
    INPUT_FILE="${INPUT_FILES[$MODEL_TYPE]}"
fi

if [ -n "$OUTPUT_DIR_ARG" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    OUTPUT_DIR="${OUTPUT_DIRS[$MODEL_TYPE]}"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Unknown model type '$MODEL_TYPE'. Use: llama, vicuna, qwen, qwen3, deepseek"
    exit 1
fi

if [ "$MODE" != "offline" ] && [ "$MODE" != "api" ]; then
    echo "Error: Unknown mode '$MODE'. Use: offline or api"
    exit 1
fi

# Parse GPU list
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "============================================"
echo "Data Generation ($MODE mode)"
echo "============================================"
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: ${GPUS[*]} ($NUM_GPUS total)"
echo "Mode: $MODE"
if [ "$MODE" == "offline" ]; then
    echo "Batch size: $BATCH_SIZE"
else
    echo "Max concurrency: $MAX_CONCURRENCY"
fi
echo "Max tokens per turn: $MAX_TOKENS"
echo "Max retries: $MAX_RETRIES"
if [ -n "$SYSTEM_PROMPT" ]; then
    echo "System prompt: ${SYSTEM_PROMPT:0:50}..."
else
    echo "System prompt: (none)"
fi
echo "============================================"

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Count total records
TOTAL_RECORDS=$(wc -l < "$INPUT_FILE")
echo "Total records: $TOTAL_RECORDS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============== Offline Mode ==============
run_offline_generation() {
    local input_file=$1
    local output_dir=$2
    local attempt=$3

    echo ""
    echo ">>> Attempt $attempt: Offline generation from $input_file"
    echo ""

    # Launch parallel processes for each GPU
    local pids=()
    for i in "${!GPUS[@]}"; do
        gpu_id="${GPUS[$i]}"
        log_file="$output_dir/gpu_${gpu_id}_attempt_${attempt}.log"

        echo "Starting shard $i on GPU $gpu_id..."

        CUDA_VISIBLE_DEVICES=$gpu_id python scripts/offline_generate_data.py \
            --model-path "$MODEL_PATH" \
            --raw-data-file "$input_file" \
            --output-dir "$output_dir" \
            --shard-id "$i" \
            --num-shards "$NUM_GPUS" \
            --batch-size "$BATCH_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            --system-prompt "$SYSTEM_PROMPT" \
            $EXTRA_FLAG \
            > "$log_file" 2>&1 &

        pids+=($!)
    done

    # Wait for all processes
    echo "Waiting for ${#pids[@]} processes..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            echo "Process $pid failed"
            ((failed++))
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "Warning: $failed processes failed"
    fi
}

# ============== API Mode ==============
run_api_generation() {
    local input_file=$1
    local output_dir=$2
    local attempt=$3

    echo ""
    echo ">>> Attempt $attempt: API generation from $input_file"
    echo ""

    # Start servers
    local server_ports=()
    local base_port=40000

    for i in "${!GPUS[@]}"; do
        gpu_id="${GPUS[$i]}"
        port=$((base_port + i))
        server_ports+=("127.0.0.1:$port")
        log_file="$output_dir/server_${gpu_id}.log"

        echo "Starting server on GPU $gpu_id, port $port..."

        CUDA_VISIBLE_DEVICES=$gpu_id python -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --port $port \
            --mem-fraction-static 0.85 \
            > "$log_file" 2>&1 &
    done

    echo "Waiting for servers to start (60s)..."
    sleep 60

    # Build server address list
    local server_args=""
    for addr in "${server_ports[@]}"; do
        server_args="$server_args $addr"
    done

    # Run generation
    echo "Starting data generation..."
    python scripts/generate_data_by_target.py \
        --model-name "$MODEL_PATH" \
        --raw-data-file "$input_file" \
        --output-dir "$output_dir" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --num-per-shard 50000 \
        --server-address-port $server_args

    # Kill servers
    echo "Stopping servers..."
    pkill -f "sglang.launch_server.*$MODEL_PATH" || true
    sleep 5
}

# ============== Common Functions ==============
collect_errors() {
    local output_dir=$1
    local retry_file=$2

    # Merge all error files
    cat "$output_dir"/error_*.jsonl 2>/dev/null | \
        python3 -c "
import sys, json
seen = set()
for line in sys.stdin:
    try:
        item = json.loads(line)
        cid = item.get('conversation_id', item.get('id', ''))
        if cid and cid not in seen:
            seen.add(cid)
            # Convert back to input format
            input_convs = item.get('input_conversations', item.get('conversations', []))
            output = {
                'id': cid,
                'conversations': input_convs
            }
            print(json.dumps(output))
    except:
        pass
" > "$retry_file"

    local count=$(wc -l < "$retry_file" 2>/dev/null || echo "0")
    echo "$count"
}

merge_results() {
    local output_dir=$1
    local final_file=$2

    echo "Merging results to $final_file..."

    # Merge all shard files (deduplicated by conversation_id)
    cat "$output_dir"/shard_*.jsonl 2>/dev/null | \
        python3 -c "
import sys, json
seen = set()
for line in sys.stdin:
    try:
        item = json.loads(line)
        cid = item.get('conversation_id', item.get('id', ''))
        if cid and cid not in seen:
            seen.add(cid)
            print(line.strip())
    except:
        pass
" > "$final_file"

    local count=$(wc -l < "$final_file")
    echo "Total records: $count"
}

# ============== Main Loop with Retry ==============
current_input="$INPUT_FILE"
attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    echo ""
    echo "============================================"
    echo "Attempt $attempt / $MAX_RETRIES"
    echo "============================================"

    if [ "$MODE" == "offline" ]; then
        run_offline_generation "$current_input" "$OUTPUT_DIR" "$attempt"
    else
        run_api_generation "$current_input" "$OUTPUT_DIR" "$attempt"
    fi

    # Collect errors
    retry_file="$OUTPUT_DIR/retry_attempt_${attempt}.jsonl"
    error_count=$(collect_errors "$OUTPUT_DIR" "$retry_file")

    echo ""
    echo "Errors in attempt $attempt: $error_count"

    if [ "$error_count" -eq 0 ] || [ "$error_count" = "0" ]; then
        echo "All records processed successfully!"
        break
    fi

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "Retrying $error_count failed records..."
        current_input="$retry_file"
    else
        echo "Max retries reached. $error_count records still failed."
    fi

    ((attempt++))
done

# ============== Final Merge ==============
echo ""
echo "============================================"
echo "Final Results"
echo "============================================"

final_output="$OUTPUT_DIR/final_output.jsonl"
merge_results "$OUTPUT_DIR" "$final_output"

# Count final errors
final_error_count=$(cat "$OUTPUT_DIR"/error_*.jsonl 2>/dev/null | wc -l || echo "0")

echo ""
echo "Summary:"
echo "  Input: $TOTAL_RECORDS records"
echo "  Success: $(wc -l < "$final_output") records"
echo "  Failed: $final_error_count records"
echo "  Output: $final_output"

if [ "$final_error_count" -gt 0 ]; then
    echo "  Error files: $OUTPUT_DIR/error_*.jsonl"
fi

echo ""
echo "Done!"
