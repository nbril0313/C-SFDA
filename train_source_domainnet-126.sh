SRC_DOMAIN=$1
GPU=$2
MODEL="conv_tiny"
PORT=$((29000 + RANDOM % 1000))
MEMO="source"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=$GPU python main_csfda.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
    data.data_root="${PWD}/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[real,sketch,clipart,painting]" \
    learn.epochs=60 \
    model_src.arch=$MODEL \
    workdir="./output1" \
    optim.lr=2e-4
done