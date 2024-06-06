SRC_MODEL_DIR="/home/ncmssh0313/C-SFDA_Source-Free-Domain-Adaptation/output/office-home/source"     ## Put the source model here
SRC_DOMAIN=$1
TGT_DOMAIN=$2

PORT=$((29000 + RANDOM % 1000))
MEMO="target"

for SEED in 2022
do
    python main_csfda_oh.py \
    seed=${SEED} port=${PORT} memo=${MEMO} project="office-home" \
    data.data_root="${PWD}/data/" data.workers=16 \
    data.dataset="office-home" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=5e-3
done
