SRC_DOMAIN=$1
GPU=$2
MODEL="resnet50"
PORT=$((19000 + RANDOM % 1000))
MEMO="source"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=$GPU python main_csfda_oh.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="office-home" \
    data.data_root="${PWD}/data" data.workers=8 \
    data.dataset="office-home" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[Art,Clipart,Product,Real_World]" \
    learn.epochs=50 \
    model_src.arch=${MODEL} \
    workdir="./output" \
    optim.lr=2e-4
done