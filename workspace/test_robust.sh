base_dir="./eval_robust_dir"
mkdir -p "${base_dir}"

GPU_ID=${GPU_ID:-0}
MODEL_NAME=${MODEL_NAME:-EARG}
TEST_DATA_PATH=${TEST_DATA_PATH:-/mnt/data0/public_datasets/IML/CASIA1.0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-./output_dir/checkpoint-0.pth}
EARG_SD_MODEL_PATH=${EARG_SD_MODEL_PATH:-/data/yuyanpu/model/SD/stable-diffusion-2-inpainting}

export EARG_SD_MODEL_PATH
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./test_robust.py \
    --model "${MODEL_NAME}" \
    --world_size 1 \
    --test_data_path "${TEST_DATA_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir "${base_dir}/" \
    --log_dir "${base_dir}/" \
    2> "${base_dir}/error.log" 1> "${base_dir}/logs.log"