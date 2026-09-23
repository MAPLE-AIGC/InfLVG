#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export WAN_ROOT="${WAN_ROOT:-../model_weights/wan_1_3b}"
export CAUSVID_CHECKPOINT="${CAUSVID_CHECKPOINT:-../model_weights/causvid_1_3b/autoregressive_checkpoint}"
export CLIP_PATH="${CLIP_PATH:-../model_weights/clip-vit-large-patch14}"
export QWEN_PATH="${QWEN_PATH:-../model_weights/qwen2_5_vl_3b}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-$HOME/.insightface}"

export total_chunk="${total_chunk:-64}"
export num_clips="${num_clips:-4}"

export epochs=1
export inner_loop="${inner_loop:-20,20,20}"
export num_gpus="${num_gpus:-5}"
export group_size="${group_size:-10}"

export use_prenorm=0
export use_postnorm=1
export opt_cls="AdamW"
export norm_type="instance" # layer
export lr=0.001
export grad_acc=1

export pm_num_layers=1
export pm_num_linear=2  

export train_start=${1:-0}
export train_end=${2:-1}

export top_n=${3:-9360} # 6*1560
export w1=1
export w2=1
export w3=1

export exp_name="${EXP_NAME:-inflvg_1_3b}"

"${PYTHON:-python}" \
    InfLVG.py \
    --config_path configs/wan_causal_dmd.yaml \
    --checkpoint_folder "$CAUSVID_CHECKPOINT"  \
    --prompt_file_path "${PROMPT_FILE:-prompts/eps_1000.jsonl}"
