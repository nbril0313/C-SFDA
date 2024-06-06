SRC_MODEL_DIR="/home/ncmssh0313/C-SFDA_Source-Free-Domain-Adaptation/output/domainnet-126/source"     ## Put the source model here
SRC_DOMAIN=$1
TGT_DOMAIN=$2

PORT=$((29000 + RANDOM % 1000))
MEMO="target"

for SEED in 2022
do
    python main_csfda.py \
    seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
    data.data_root="${PWD}/data/" data.workers=4 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=5e-4
done
