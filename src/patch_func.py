from typing import List, Union, Dict

import torch
import pickle
import sys

from nanotron.parallel.pipeline_parallel.block import TensorPointer


def create_custom_apply_rotary_pos_emb(cfg):
    def apply_rotary_pos_emb_v0(self, q, k, cos, sin, unsqueeze_dim=2):
        # Full-RoPE
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v1(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the fastest-rotating (high-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_dim = cfg["top_k_rope_dim"]
        if keep_dim <= 0:
            return q, k
        elif keep_dim >= q.size(-1):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., :keep_dim],
                q[..., keep_dim:half],
                q_embed[..., half : half + keep_dim],
                q[..., half + keep_dim :],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :keep_dim],
                k[..., keep_dim:half],
                k_embed[..., half : half + keep_dim],
                k[..., half + keep_dim :],
            ),
            -1,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v2(self, q, k, cos, sin, unsqueeze_dim=2):
        # select subspaces with equidistant intervals
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        indices = torch.arange(
            cfg["uniform_start_point"], q.size(-1), cfg["uniform_step"], device=q.device
        )
        q,k = q.clone(),k.clone()
        q[..., indices] = q_embed[..., indices]
        k[..., indices] = k_embed[..., indices]
        return q, k

    def apply_rotary_pos_emb_v3(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the fastest-rotating (high-frequency) subspaces and the slowest-rotating (low-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        top_k_dim, last_k_dim = cfg["top_k_rope_dim"], cfg["last_k_rope_dim"]
        # assert top_k_dim + last_k_dim <= q.size(-1)
        half = q.size(-1) // 2
        qs = [
            q_embed[..., :top_k_dim],
            q[..., top_k_dim : half - last_k_dim],
            q_embed[..., half - last_k_dim : half + top_k_dim],
            q[..., half + top_k_dim : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k_embed[..., :top_k_dim],
            k[..., top_k_dim : half - last_k_dim],
            k_embed[..., half - last_k_dim : half + top_k_dim],
            k[..., half + top_k_dim : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v4(self, q, k, cos, sin, layer_idx=0, unsqueeze_dim=2):
        # retain the subspaces with higher 2-norm score
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        top_k_dim = cfg["top_k_rope_dim"]
        qk_tensor[layer_idx] = qk_tensor[layer_idx].to(q.device)
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        mask = torch.zeros_like(qk_tensor[layer_idx])
        mask.scatter_(1, topk_indices, 1)
        mask_for_k = (
            torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(1).to(q.device)
        )
        mask_for_q = torch.repeat_interleave(
            input=mask_for_k, repeats=cfg["n_gqa_group"], dim=2
        ).to(q.device)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v5(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the slowest-rotating (low-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        last_k_dim = cfg["last_k_rope_dim"]
        half = q.size(-1) // 2
        qs = [
            q[..., : half - last_k_dim],
            q_embed[..., half - last_k_dim : half],
            q[..., half : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k[..., : half - last_k_dim],
            k_embed[..., half - last_k_dim : half],
            k[..., half : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed
    
    def apply_rotary_pos_emb_v6(self, q, k, cos, sin, layer_idx=0, unsqueeze_dim=2):
        # retain the subspaces with higher 2-norm score and different head could have different r
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        mask = mask_tensor[layer_idx].to(q.device)
        mask_for_k = (
            torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(1).to(q.device)
        )
        mask_for_q = torch.repeat_interleave(
            input=mask_for_k, repeats=cfg["n_gqa_group"], dim=2
        ).to(q.device)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    version = cfg["partial_rope_version"]
    if version == 4 or version == 6:
        with open(cfg["qk_tensor_path"], "rb") as fin:
            qk_tensor = torch.load(fin).cuda()
    if version == 6:
        flattened_qk = qk_tensor.view(qk_tensor.size(0), -1)
        total_dim = qk_tensor.size(1) * cfg["top_k_rope_dim"]
        _, top_indices = torch.topk(flattened_qk, total_dim, dim=1)
        mask_tensor = torch.zeros_like(flattened_qk, dtype=torch.bool)
        mask_tensor.scatter_(1, top_indices, 1)
        mask_tensor = mask_tensor.view_as(qk_tensor)

    versions = {
        0: apply_rotary_pos_emb_v0,
        1: apply_rotary_pos_emb_v1,
        2: apply_rotary_pos_emb_v2,
        3: apply_rotary_pos_emb_v3,
        4: apply_rotary_pos_emb_v4,
        5: apply_rotary_pos_emb_v5,
        6: apply_rotary_pos_emb_v6,
    }
    return versions.get(version, apply_rotary_pos_emb_v0)


def custom_forward_with_hidden_states_for_v4(
    self,
    input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
):
    # all tensors are optional as most ranks don't need anything from the dataloader.

    output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

    hidden_encoder_states = {
        "hidden_states": output["input_embeds"],
        "sequence_mask": input_mask,
    }
    # solve module_input_keys not match error
    if "layer_idx" not in self.decoder[0].module_input_keys:
        for layer_idx, encoder_block in enumerate(self.decoder):
            encoder_block.module_input_keys.add("layer_idx")
    for layer_idx, encoder_block in enumerate(self.decoder):
        hidden_encoder_states = encoder_block(
            **hidden_encoder_states, layer_idx=layer_idx
        )

    hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])[
        "hidden_states"
    ]

    sharded_logits = self.lm_head(x=hidden_states)["logits"]

    fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

    return fp32_sharded_logits, hidden_states


def custom_decoder_forward_for_v4(
    self,
    hidden_states: Union[torch.Tensor, TensorPointer],
    sequence_mask: Union[torch.Tensor, TensorPointer],
    layer_idx: int,
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
        hidden_states, sequence_mask = self._checkpointed_forward(
            hidden_states, sequence_mask
        )
    else:
        hidden_states, sequence_mask = self._core_forward(
            hidden_states, sequence_mask, layer_idx
        )

    return {
        "hidden_states": hidden_states,
        "sequence_mask": sequence_mask,
    }


def custom_decoder_core_forward_for_v4(
    self,
    hidden_states: Union[torch.Tensor, TensorPointer],
    sequence_mask: Union[torch.Tensor, TensorPointer],
    layer_idx: int,
) -> List[Union[torch.Tensor, TensorPointer]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    output = self.attn(
        hidden_states=hidden_states, sequence_mask=sequence_mask, layer_idx=layer_idx
    )
    hidden_states = output["hidden_states"]
    hidden_states = hidden_states + residual

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
    hidden_states = hidden_states + residual

    return hidden_states, output["sequence_mask"]


def custom_llama_forward_for_v4(
    self,
    hidden_states,  # [seq_length, batch_size, hidden_size]
    sequence_mask,  # [batch_size, seq_length]
    layer_idx,
):
    from flash_attn import bert_padding
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    qkv_states = self.qkv_proj(
        hidden_states
    )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
    q_length, batch_size, _ = qkv_states.shape

    if self.is_gqa:
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.n_local_q_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
            ],
            dim=-1,
        )

        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )
        key_states = (
            key_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
        value_states = (
            value_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
    else:
        query_states, key_states, value_states = (
            qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

    store = self.get_local_store()
    if store is not None:  # Inference case
        # Double check that we use store only at inference time
        assert key_states.requires_grad is False
        assert value_states.requires_grad is False
        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        # Compute rotary embeddings
        # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
        old_rotary_embed_end = self.rotary_embedding.end
        # interleaved version.
        if self.rope_interleaved:
            query_states = self.rotary_embedding(
                query_states, position_ids=position_ids
            )
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        # non interleaved version.
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if "key" not in store:
            # First inference iteration (Prefill)
            # TODO @nouamane: support custom masking
            # assert that [ False, False, False, False,  True,  True,  True,  True,  True,  True] is accepted
            # but [ False, False, False, False,  True,  True,  False,  False,  True,  True] is not (can't mask in the middle of sequence)
            assert ~(
                sequence_mask[:, :-1]
                & (~sequence_mask[:, 1:])  # True is never followed by False
            ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

            # preallocate k_cache, v_cache to self.prefill_kv_len
            k_cache = torch.zeros(
                (
                    batch_size,
                    self.prefill_kv_len,
                    self.n_local_kv_heads,
                    self.d_qk,
                ),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            v_cache = torch.zeros(
                (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            # Remove pad tokens from key_states and concatenate samples in key_unpad
            # cu_seqlens_k is the cumulative sequence lengths of key_states
            (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = (
                bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
            )
            (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = (
                bert_padding.unpad_input(key_states, sequence_mask)
            )
            (value_unpad, _, _, _) = bert_padding.unpad_input(
                value_states, sequence_mask
            )

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            output_unpad = flash_attn_varlen_func(
                q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,  # True in prefill phase, False in subsequent phases
                return_attn_probs=False,
            )  # (total_unpadded, n_local_q_heads, d_v)

            attention_output = bert_padding.pad_input(
                output_unpad, indices_q, batch_size, q_length
            )  # (batch_size, q_length, n_local_q_heads, d_v)

            pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
            pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

        else:
            # Pull pre-computed key/value states
            # Subsequent inference iterations (q_length=1)
            k_cache = store["key"]
            v_cache = store["value"]

            # NOTE(fmom): According to flash_attn_with_kvcache, "If you pass in k / v, you must make sure that the cache is large enough to hold the new values"
            # Since rotary embedding has changed (to enable larger context), we need to enlarge k_cache and v_cache
            if self.rotary_embedding.end > old_rotary_embed_end:
                k_cache = torch.cat(
                    [
                        k_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_qk,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

                v_cache = torch.cat(
                    [
                        v_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_v,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

            assert (
                k_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
            assert (
                v_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

            # [batch_size, seq_length, num_heads, d_qk]
            query_states = query_states.view(
                batch_size, q_length, self.n_local_q_heads, self.d_qk
            )  # [batch_size, q_length, self.n_heads, d_qk]
            kv_length = key_states.shape[1]
            key_states = key_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_qk
            )  # [batch_size, kv_length, self.n_heads, d_qk]
            value_states = value_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_v
            )  # [batch_size, kv_length, self.n_heads, d_v]

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            attention_output = flash_attn_with_kvcache(
                query_states,
                k_cache,
                v_cache,
                key_states,
                value_states,
                rotary_cos=None,
                rotary_sin=None,
                # TODO @nouamane: seems like this doesn't help to indicate padding in (for first iteration it's just 0)
                cache_seqlens=position_offsets.contiguous(),
                softmax_scale=softmax_scale,
                causal=True,
                rotary_interleaved=False,  # the value is not used unless rotary_cos/sin is provided. https://github.com/Dao-AILab/flash-attention
            )

        store.update(
            {
                "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                "value": v_cache,
                "position_offsets": position_offsets,
            }
        )

    else:  # Training case
        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # # [2, batch_size, seq_length, num_heads, d_qk]
        # key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # # [batch_size, seq_length, 2, num_heads, d_qk]
        # key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        # query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        # # [batch_size, seq_length, num_heads, d_qk]
        # key_states, value_states = torch.split(key_value_states, 1, dim=2)
        position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        cos, sin = self.rotary_embedding(value_states, position_ids)
        query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, layer_idx
        )
        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

    attention_output = (
        attention_output.contiguous()
        .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
        .transpose(0, 1)
    )
    output = self.o_proj(attention_output)

    return {"hidden_states": output, "sequence_mask": sequence_mask}


def custom_llama_forward(
    self,
    hidden_states,  # [seq_length, batch_size, hidden_size]
    sequence_mask,  # [batch_size, seq_length]
):
    from flash_attn import bert_padding
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    qkv_states = self.qkv_proj(
        hidden_states
    )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
    q_length, batch_size, _ = qkv_states.shape

    if self.is_gqa:
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.n_local_q_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
            ],
            dim=-1,
        )

        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )
        key_states = (
            key_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
        value_states = (
            value_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
    else:
        query_states, key_states, value_states = (
            qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

    store = self.get_local_store()
    if store is not None:  # Inference case
        # Double check that we use store only at inference time
        assert key_states.requires_grad is False
        assert value_states.requires_grad is False
        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        # Compute rotary embeddings
        # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
        old_rotary_embed_end = self.rotary_embedding.end
        # interleaved version.
        if self.rope_interleaved:
            query_states = self.rotary_embedding(
                query_states, position_ids=position_ids
            )
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        # non interleaved version.
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if "key" not in store:
            # First inference iteration (Prefill)
            # TODO @nouamane: support custom masking
            # assert that [ False, False, False, False,  True,  True,  True,  True,  True,  True] is accepted
            # but [ False, False, False, False,  True,  True,  False,  False,  True,  True] is not (can't mask in the middle of sequence)
            assert ~(
                sequence_mask[:, :-1]
                & (~sequence_mask[:, 1:])  # True is never followed by False
            ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

            # preallocate k_cache, v_cache to self.prefill_kv_len
            k_cache = torch.zeros(
                (
                    batch_size,
                    self.prefill_kv_len,
                    self.n_local_kv_heads,
                    self.d_qk,
                ),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            v_cache = torch.zeros(
                (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            # Remove pad tokens from key_states and concatenate samples in key_unpad
            # cu_seqlens_k is the cumulative sequence lengths of key_states
            (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = (
                bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
            )
            (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = (
                bert_padding.unpad_input(key_states, sequence_mask)
            )
            (value_unpad, _, _, _) = bert_padding.unpad_input(
                value_states, sequence_mask
            )

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            output_unpad = flash_attn_varlen_func(
                q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,  # True in prefill phase, False in subsequent phases
                return_attn_probs=False,
            )  # (total_unpadded, n_local_q_heads, d_v)

            attention_output = bert_padding.pad_input(
                output_unpad, indices_q, batch_size, q_length
            )  # (batch_size, q_length, n_local_q_heads, d_v)

            pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
            pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

        else:
            # Pull pre-computed key/value states
            # Subsequent inference iterations (q_length=1)
            k_cache = store["key"]
            v_cache = store["value"]

            # NOTE(fmom): According to flash_attn_with_kvcache, "If you pass in k / v, you must make sure that the cache is large enough to hold the new values"
            # Since rotary embedding has changed (to enable larger context), we need to enlarge k_cache and v_cache
            if self.rotary_embedding.end > old_rotary_embed_end:
                k_cache = torch.cat(
                    [
                        k_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_qk,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

                v_cache = torch.cat(
                    [
                        v_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_v,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

            assert (
                k_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
            assert (
                v_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

            # [batch_size, seq_length, num_heads, d_qk]
            query_states = query_states.view(
                batch_size, q_length, self.n_local_q_heads, self.d_qk
            )  # [batch_size, q_length, self.n_heads, d_qk]
            kv_length = key_states.shape[1]
            key_states = key_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_qk
            )  # [batch_size, kv_length, self.n_heads, d_qk]
            value_states = value_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_v
            )  # [batch_size, kv_length, self.n_heads, d_v]

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            attention_output = flash_attn_with_kvcache(
                query_states,
                k_cache,
                v_cache,
                key_states,
                value_states,
                rotary_cos=None,
                rotary_sin=None,
                # TODO @nouamane: seems like this doesn't help to indicate padding in (for first iteration it's just 0)
                cache_seqlens=position_offsets.contiguous(),
                softmax_scale=softmax_scale,
                causal=True,
                rotary_interleaved=False,  # the value is not used unless rotary_cos/sin is provided. https://github.com/Dao-AILab/flash-attention
            )

        store.update(
            {
                "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                "value": v_cache,
                "position_offsets": position_offsets,
            }
        )

    else:  # Training case
        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # # [2, batch_size, seq_length, num_heads, d_qk]
        # key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # # [batch_size, seq_length, 2, num_heads, d_qk]
        # key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        # query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        # # [batch_size, seq_length, num_heads, d_qk]
        # key_states, value_states = torch.split(key_value_states, 1, dim=2)
        position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        cos, sin = self.rotary_embedding(value_states, position_ids)
        query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

    attention_output = (
        attention_output.contiguous()
        .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
        .transpose(0, 1)
    )
    output = self.o_proj(attention_output)

    return {"hidden_states": output, "sequence_mask": sequence_mask}


def pad_to_right(tensor, mask, new_tensor=None):
    """Transform a left-padded tensor into a right-padded tensor. (Useful for prefilling key/value states)
    Args:
        tensor: (batch_size, seqlen, d1, d2)
        mask: (batch_size, seqlen)
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
    Returns:
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
        right_padded_mask: (batch_size, seqlen)
    """
    # First, we need to find the number of padding for each row
    unpad_seqlens = mask.sum(1)
    # Then, we need to find the maximum length of the tensor
    max_seqlen = mask.shape[1]
    # We can then create the indices to select the padded values
    # The indices are the same for each row
    indices = torch.arange(max_seqlen, device=mask.device)
    # We can then create the mask for the padded values
    right_padded_mask = indices < unpad_seqlens[:, None]
    # We select the useful values
    useful_values = tensor[mask]
    # We create the new tensor (if not provided)
    new_tensor = torch.zeros_like(tensor) if new_tensor is None else new_tensor
    # We fill the new tensor with the useful values
    new_tensor[:, : right_padded_mask.shape[1], :, :][right_padded_mask] = useful_values
    return new_tensor, right_padded_mask
