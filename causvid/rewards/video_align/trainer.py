from typing import List, Optional, Dict, Union, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration

from transformers.trainer import TrainerCallback


IS_XLA_FSDPV2_POST_2_2 = False

class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
        super().__init__(config)
        # pdb.set_trace()
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        input_ids = input_ids.to(self.model.device)
        pixel_values_videos = pixel_values_videos.to(self.model.device, torch.bfloat16)

        ## modified from the origin class Qwen2VLForConditionalGeneration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]  # [B, L, D]

        logits = self.rm_head(hidden_states)    # [B, L, N]
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        ## get sequence length
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        ## get the last token's logits
        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            ## get the mean of all valid tokens' logits
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            # special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
            # create a mask for special tokens
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)   # [B, 3, N] assert 3 attributes
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)

            # pdb.set_trace()
        else:
            raise ValueError("Invalid reward_token")
        
        return {"logits": pooled_logits}



class PartialEmbeddingUpdateCallback(TrainerCallback):
    """
    Callback to update the embedding of special tokens
    Only the special tokens are updated, the rest of the embeddings are kept fixed
    """
    def __init__(self, special_token_ids):
        super().__init__()
        self.special_token_ids = special_token_ids
        self.orig_embeds_params = None 

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self.orig_embeds_params = model.get_input_embeddings().weight.clone().detach()

    def on_step_end(self, args, state, control, **kwargs):
        # pdb.set_trace()
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[self.special_token_ids] = False
        with torch.no_grad():
            model.get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[index_no_updates]
            

def compute_multi_attr_accuracy(eval_pred, metainfo_idxs=None, eval_dims=None, save_path=None) -> Dict[str, float]:
    predictions, labels = eval_pred
    metrics = {}
    for idx, eval_dim in enumerate(eval_dims):
        pred_curr = predictions[:, :, idx]
        label_curr = labels[:, idx]
        # pdb.set_trace()
        ## calculate the average scores of rewards_chosen and rewards_rejected
        valid_mask = (label_curr != 0)

        rewards_chosen = np.where(label_curr == 1, pred_curr[:, 0], pred_curr[:, 1])
        rewards_rejected = np.where(label_curr == -1, pred_curr[:, 0], pred_curr[:, 1])

        rewards_chosen_avg = np.sum(rewards_chosen * valid_mask) / np.sum(valid_mask)
        rewards_rejected_avg = np.sum(rewards_rejected * valid_mask) / np.sum(valid_mask)

        pred_curr = np.argmax(pred_curr, axis=1)
        pred_curr = np.where(pred_curr == 0, 1, -1)
        accuracy = np.array(pred_curr == label_curr, dtype=float)
        accuracy = np.sum(accuracy * valid_mask) / np.sum(valid_mask)

        metrics.update({
            f"accuracy_{eval_dim}": accuracy,
            f"rewards_chosen_avg_{eval_dim}": rewards_chosen_avg,
            f"rewards_rejected_avg_{eval_dim}": rewards_rejected_avg,
        })

    if save_path is not None and metainfo_idxs is not None:
        df = pd.DataFrame(metainfo_idxs, columns=["metainfo_idx"])
        for idx, eval_dim in enumerate(eval_dims):
            rewards_A = predictions[:, 0, idx]
            rewards_B = predictions[:, 1, idx]
            df[f"reward_A_{eval_dim}"] = rewards_A
            df[f"reward_B_{eval_dim}"] = rewards_B
        
        df.to_csv(save_path, index=False)
        print(f"===> Inference results saved to {save_path}")

    # pdb.set_trace()
    return metrics