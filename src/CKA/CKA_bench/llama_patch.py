import sys
sys.path.append("../../../../xKV/")
from xKV.attn_patch.llama import xKV_llama_sdpa_forward
from transformers.models.llama.modeling_llama import (
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaForCausalLM,
    LlamaModel
)
from transformers.generation.utils import GenerationMixin
from typing import Dict, Optional, Union, Tuple, List
from transformers.cache_utils import Cache, DynamicCache
from xKV.customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache
from loguru import logger
xkv_config = None

from xKV.configurations import xKVConfig
def get_xkv_config(yaml_path):
    config = xKVConfig.from_yaml(yaml_path)
    return config

def _prepare_cache_for_generation(
    self, generation_config, model_kwargs: Dict, *args, **kwargs
) -> bool:
    """
    Prepares the cache for generation (if applicable), given `generate`'s paramaterization. If a cache is
    instantiated, writes it to `model_kwargs`, under the name expected by the model.
    """
    if xkv_config is None:
        raise ValueError("xkv config is None")
    model_kwargs["past_key_values"] = FakeLayerMergingCache(xkv_config)
    logger.info("Cache obj: FakeLayerMergingCache")

def enable_llama_xKV_patch_generate(yaml_path):
    LlamaSdpaAttention.forward = xKV_llama_sdpa_forward
    GenerationMixin._prepare_cache_for_generation = _prepare_cache_for_generation
    global xkv_config
    xkv_config = get_xkv_config(yaml_path)

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            global xkv_config
            if xkv_config is None:
                raise ValueError("xkv config is None")
            past_key_values = FakeLayerMergingCache(xkv_config)
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def enable_llama_xKV_patch_forward(yaml_path):
    LlamaSdpaAttention.forward = xKV_llama_sdpa_forward
    global xkv_config
    xkv_config = get_xkv_config(yaml_path)
    LlamaModel.forward = forward
