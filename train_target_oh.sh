SRC_MODEL_DIR="/home/ncmssh0313/C-SFDA_Source-Free-Domain-Adaptation/output/office-home/source"     ## Put the source model path here
SRC_DOMAINS=("Art" "Art" "Art" "Clipart" "Clipart" "Clipart" "Product" "Product" "Product" "Real_World" "Real_World" "Real_World")
TGT_DOMAINS=("Clipart" "Product" "Real_World" "Art" "Product" "Real_World" "Art" "Clipart" "Real_World" "Art" "Clipart" "Product")
GPU=$1
# Art,Clipart,Product,Real_World
for i in "${!SRC_DOMAINS[@]}"; do
    SRC_DOMAIN=${SRC_DOMAINS[$i]}
    TGT_DOMAIN=${TGT_DOMAINS[$i]}
    PORT=$((19000 + RANDOM % 1000))
    MEMO="target"

    for SEED in 2022; do
        CUDA_VISIBLE_DEVICES=$GPU python main_csfda_oh.py \
        seed=${SEED} port=${PORT} memo=${MEMO} project="office-home" \
        data.data_root="${PWD}/data/" data.workers=4 \
        data.dataset="office-home" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
        model_src.arch="resnet50" \
        data.batch_size=128 \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        optim.lr=5e-3
    done
done