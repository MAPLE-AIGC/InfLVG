import json
import os
from dataclasses import asdict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from trl import get_kbit_device_map, get_quantization_config

from .trainer import Qwen2VLRewardModelBT
from .data import convert_GSB_csv_to_reward_data


# def save_configs_to_json(data_config, training_args, model_config, peft_lora_config):
#     """
#     Save all configurations to a JSON file.
#     """
#     config_dict = {
#         "data_config": asdict(data_config),
#         "training_args": asdict(training_args),
#         "model_config": asdict(model_config),
#         "peft_lora_config": asdict(peft_lora_config),
#     }
#     # del information about local device
#     del config_dict["training_args"]["local_rank"]
#     del config_dict["training_args"]["_n_gpu"]

#     save_path = os.path.join(training_args.output_dir, "model_config.json")

#     os.makedirs(training_args.output_dir, exist_ok=True)
#     print(training_args.output_dir)

#     with open(save_path, "w") as f:
#         json.dump(config_dict, f, indent=4)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    """
    Find the target linear modules for LoRA.
    """
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            # print(f"Excluding module: {name}")
            continue

        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def create_model_and_processor(
        model_config, peft_lora_config, training_args,
        cache_dir=None,
    ):
    # create model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True, #  if training_args.gradient_checkpointing else False,
    )
    # pdb.set_trace()

    # create processor and set padding
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path,
                                              padding_side="right",
                                              cache_dir=cache_dir)
    
    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2", # if not training_args.disable_flash_attn2 else "sdpa",
        cache_dir=cache_dir,
        **model_kwargs
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer)) 

    # if training_args.bf16:
    model.to(torch.bfloat16)
    # if training_args.fp16:
    #     model.to(torch.float16)

    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude)
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config

def create_dataset(data_config, meta_file=None):
    if meta_file is None:
        meta_file = data_config.meta_data
    dataset = load_dataset('csv', data_files=meta_file)
    def add_idx(example, idx):
        example['metainfo_idx'] = idx
        return example
    dataset['train'] = dataset['train'].map(lambda example, idx: add_idx(example, idx), with_indices=True) 
    
    if not data_config.use_tied_data:
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in data_config.eval_dim)
        dataset = dataset.filter(filter_func)

    # convert data to reward data
    convert_func = lambda example: convert_GSB_csv_to_reward_data(example, data_config.data_dir, data_config.eval_dim, 
                                                                  data_config.max_frame_pixels, data_config.fps, data_config.num_frames,
                                                                  data_config.prompt_template_type,
                                                                  sample_type=data_config.sample_type,)
    dataset = dataset.map(convert_func, remove_columns=dataset['train'].column_names, load_from_cache_file=False)
    dataset = dataset['train']
    # pdb.set_trace()
    return dataset
