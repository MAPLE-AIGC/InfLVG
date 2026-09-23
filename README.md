<div align="center">

# InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO

Xueji Fang, Liyuan Ma&#42;, Zhiyang Chen, Mingyuan Zhou, Guo-jun Qi&#42;

MAPLE Lab, Westlake University

&#42; Corresponding authors

[Paper](https://arxiv.org/abs/2505.17574)

[Overview](#overview) · [Getting Started](#getting-started) · [Generation](#generation) · [Event Prompts](#event-prompts)

</div>

## Overview

InfLVG generates a long video from a sequence of scene prompts while preserving the subject across scene changes. At each transition, it trains a context-selection policy online with Group Relative Policy Optimization (GRPO). The CausVid generator remains frozen; no pretrained InfLVG policy checkpoint is needed.

This release uses the **Wan2.1 T2V 1.3B CausVid** generator. It includes the online policy, Qwen2.5-VL, CLIP, and InsightFace rewards, and Event Prompt Sets (EPS). Model weights are downloaded separately.

## Getting Started

### Installation

Create a Python 3.11 environment, install a CUDA-enabled PyTorch and matching torchvision from the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
git clone https://github.com/MAPLE-AIGC/InfLVG.git
cd InfLVG_cs
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The example was verified with PyTorch 2.10, Diffusers 0.37, Transformers 5.3, and eight 80 GB GPUs. The script places five generation workers on GPUs 0–4 and long-history KV caches on GPUs 5–7. Reward models share GPU 1.

### Model weights

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) with `python -m pip install -U huggingface_hub`. Download the generator, Wan text encoder and VAE, CLIP, and Qwen2.5-VL:

```bash
hf download tianweiy/CausVid autoregressive_checkpoint/model.pt \
  --local-dir ../model_weights/causvid_1_3b
hf download Wan-AI/Wan2.1-T2V-1.3B \
  config.json Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir ../model_weights/wan_1_3b
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" --local-dir ../model_weights/wan_1_3b
hf download openai/clip-vit-large-patch14 \
  config.json preprocessor_config.json pytorch_model.bin merges.txt \
  vocab.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir ../model_weights/clip-vit-large-patch14
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ../model_weights/qwen2_5_vl_3b
```

Download the `antelopev2` pack from the [InsightFace model zoo](https://github.com/deepinsight/insightface/releases/tag/model-zoo) and extract its ONNX files to `~/.insightface/models/antelopev2/`. InsightFace model weights are provided for noncommercial research use.

`run.sh` uses these locations by default. Set `WAN_ROOT`, `CAUSVID_CHECKPOINT`, `CLIP_PATH`, `QWEN_PATH`, or `INSIGHTFACE_ROOT` to use other local paths. `CAUSVID_CHECKPOINT` points to the directory containing `model.pt`; `INSIGHTFACE_ROOT` points to the parent of `models/antelopev2`.

## Generation

Run the first EPS with the four-scene configuration:

```bash
bash run.sh 0 1
```

The defaults are 64 three-frame latent blocks (765 output frames), four scenes, 20 policy updates per transition, group size 10, `K=9360` retained context tokens, and learning rate `0.001`. The script reads `prompts/eps_1000.jsonl` and writes videos to `logging/inflvg_1_3b/`. To process all 1,000 EPS rows, use `bash run.sh 0 1000`.

For the verified, shorter two-scene example:

```bash
total_chunk=32 num_clips=2 inner_loop=1,1 group_size=10 num_gpus=5 \
PROMPT_FILE=prompts/demo.jsonl EXP_NAME=demo bash run.sh 0 1
```

Its complete video is written to `logging/demo/aa_epoch_0_1_clean_clip1.mp4`. Generation can vary because candidate sampling is stochastic.

## Event Prompts

`prompts/eps_1000.jsonl` contains 1,000 Event Prompt Sets, selected as the first 1,000 rows of the original EPS v4 file. Each row has six numbered scene prompts; the default run uses the first four. `prompts/demo.jsonl` contains the single example used for the short command.

## Acknowledgements

The generator builds on [CausVid](https://github.com/tianweiy/CausVid) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). The reward models come from [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [CLIP](https://huggingface.co/openai/clip-vit-large-patch14), and [InsightFace](https://github.com/deepinsight/insightface). Check each model's terms before using its weights.

## Citation

```bibtex
@misc{fang2025inflvg,
  title={InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO},
  author={Xueji Fang and Liyuan Ma and Zhiyang Chen and Mingyuan Zhou and Guo-jun Qi},
  year={2025},
  eprint={2505.17574},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.17574}
}
```
