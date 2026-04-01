base_dir="./eval_dir"
mkdir -p "${base_dir}"

GPU_ID=${GPU_ID:-0}
MODEL_NAME=${MODEL_NAME:-EARG}
TEST_DATA_JSON=${TEST_DATA_JSON:-./test_datasets.json}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-./output_dir/}
EARG_SD_MODEL_PATH=${EARG_SD_MODEL_PATH:-/data/model/stable-diffusion-2-inpainting}

export EARG_SD_MODEL_PATH
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./test.py \
    --model "${MODEL_NAME}" \
    --world_size 1 \
    --test_data_json "${TEST_DATA_JSON}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir "${base_dir}/" \
    --log_dir "${base_dir}/" \
    2> "${base_dir}/error.log" 1> "${base_dir}/logs.log"