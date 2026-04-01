base_dir="./output_dir"
mkdir -p "${base_dir}"

GPU_ID=${GPU_ID:-0}
MODEL_NAME=${MODEL_NAME:-EARG}
DATA_PATH=${DATA_PATH:-./balanced_dataset.json}
TEST_DATA_PATH=${TEST_DATA_PATH:-./validation_dataset.json}
EARG_SD_MODEL_PATH=${EARG_SD_MODEL_PATH:-/data/yuyanpu/model/SD/stable-diffusion-2-inpainting}

export EARG_SD_MODEL_PATH
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./train.py \
    --model "${MODEL_NAME}" \
    --world_size 1 \
    --batch_size 8 \
    --data_path "${DATA_PATH}" \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --find_unused_parameters \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "${TEST_DATA_PATH}" \
    --warmup_epochs 2 \
    --output_dir "${base_dir}/" \
    --log_dir "${base_dir}/" \
    --accum_iter 8 \
    --seed 42 \
    --test_period 10 \
    2> "${base_dir}/error.log" 1> "${base_dir}/logs.log"
