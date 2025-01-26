from typing import Any, Dict, Optional
import nanotron.config
import nanotron.config
import nanotron.config.config
import nanotron.config.models_config
import nanotron.trainer
import torch
from torch import nn
import torch.distributed as dist
from packaging.version import Version
from pathlib import Path
from dataclasses import dataclass, field
import nanotron
from nanotron.s3_checkpoints import check_path_is_local
from safetensors.torch import safe_open
from tqdm import tqdm
from nanotron.config import (
    ParallelismArgs,
    LlamaConfig,
    ExistingCheckpointInit,
    RandomInit,
    SpectralMupInit,
)
from nanotron.generation.generate_store import AttachableStore
from torch.nn.parallel import DistributedDataParallel
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.logging import log_rank
from nanotron import logging
from nanotron.distributed import get_global_rank
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.models import NanotronModel
from nanotron.models import llama
from nanotron.models.llama import (
    CoreAttention,
    logger,
)
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.parallel.parameters import NanotronParameter
from nanotron.serialize import weights as nt_weights
from nanotron.serialize.weights import (
    get_checkpoint_version,
    read_checkpoint_version_from_shard_file,
    CheckpointVersionFromShardFileException,
    load_sharded_param_latest,
)
from nanotron.parallel import ParallelContext
from nanotron.serialize.utils import (
    ObjectType,
    get_exp_tp_pp_rank_and_size_from,
    get_path,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ..mla.NopeIndex import IndexForNope
from ..mla.svd_low_rank import SvdInit


@dataclass
class CustomLlamaConfig(LlamaConfig):
    RoPE: Dict = field(default_factory=dict)
    SVD: Dict = field(default_factory=dict)


@dataclass
class CustomModelArgs(nanotron.config.config.ModelArgs):
    model_config: CustomLlamaConfig


@dataclass
class CustomConfig(nanotron.config.config.Config):
    """Main configuration class"""

    model: CustomModelArgs


class CustomCausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        assert (
            not config.rope_interleaved
        ), "MLA Causal attention does not support interleaved RoPE"
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = (
            config.num_attention_heads != config.num_key_value_heads
        )  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup
        self.layer_idx = layer_idx
        self.is_share_W_down = bool(config.SVD["method"] not in [2, 3])
        self.low_rank = config.SVD["low_rank"]
        self.tp_mode = (
            self.parallel_config.tp_mode
            if self.parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.tp_linear_async_communication = (
            self.parallel_config.tp_linear_async_communication
            if self.parallel_config is not None
            else False
        )
        # TODO @thomasw21: refactor so that we store that default in a single place.

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk

        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        self.rotary_embedding = llama.LlamaRotaryEmbedding(
            dim=self.d_qk,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.rope_interleaved = config.rope_interleaved
        self.init_w_q()
        self.init_w_k()
        self.init_w_v()
        self.init_w_o()

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.max_position_embeddings
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass max_position_embeddings


    def init_w_q(self):
        self.q_proj = TensorParallelColumnLinear(
            self.d_model,
            self.config.num_attention_heads * self.d_qk,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def init_w_k(self):
        self.k_proj = TensorParallelColumnLinear(
            self.d_model,
            self.n_local_kv_heads * self.d_qk,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        # self.W_down_k = TensorParallelColumnLinear(
        #     self.d_model,
        #     self.low_rank * self.n_local_kv_heads,
        #     bias=False,
        #     pg=self.tp_pg,
        #     mode=self.tp_mode,
        #     async_communication=self.tp_linear_async_communication,
        # )
        # self.W_up_k = TensorParallelColumnLinear(
        #     self.low_rank * self.n_local_kv_heads,
        #     self.n_local_kv_heads * self.d_qk - (self.nope_mask == False).sum().item(),
        #     bias=False,
        #     pg=self.tp_pg,
        #     mode=self.tp_mode,
        #     async_communication=self.tp_linear_async_communication,
        # )

    def init_w_v(self):
        self.v_proj = TensorParallelColumnLinear(
            self.d_model,
            self.n_local_kv_heads * self.d_v,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_down_v = TensorParallelColumnLinear(
            self.d_model,
            self.low_rank * self.n_local_kv_heads,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
        )
        self.W_up_v = TensorParallelColumnLinear(
            self.low_rank * self.n_local_kv_heads,
            self.d_model,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
        )
        # self.W_up_v = TensorParallelColumnLinear(
        #     self.low_rank * self.n_local_kv_heads,
        #     self.n_local_kv_heads * self.d_v,
        #     bias=False,
        #     pg=self.tp_pg,
        #     mode=self.tp_mode,
        #     async_communication=self.tp_linear_async_communication,
        # )

    def init_w_o(self):
        self.o_proj = TensorParallelRowLinear(
            self.config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
        )

    def get_query_states(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        query_states = self.q_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk]
        q_length, batch_size, _ = query_states.shape
        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )  # [batch_size, seq_length, n_local_q_heads, d_qk]
        return query_states

    def get_key_states(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        key_states = self.k_proj(hidden_states)  # [seq_length, batch_size, n_local_kv_heads * d_qk]
        # k_r = self.W_k_r(hidden_states)
        # k_c = self.W_up_k(self.W_down_k(hidden_states))
        # key_states = torch.zeros(
        #     (hidden_states.size(0),hidden_states.size(1),self.n_local_kv_heads*self.d_qk),
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device,
        # )
        # key_states[...,~self.nope_mask] = k_r
        # key_states[...,self.nope_mask] = k_c
        key_states = (
            key_states.transpose(0,1)
            .contiguous()
            .view(hidden_states.size(1),hidden_states.size(0),self.n_local_kv_heads,self.d_qk)
        )
        return key_states
    
    def get_value_states(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        # value_states = self.v_proj(hidden_states)  # [seq_length, batch_size, n_local_kv_heads * d_v]
        # c_k = self.W_down_v(hidden_states)
        # value_states = self.W_up_v(c_k)
        value_states = self.W_down_v(hidden_states)
        value_states = self.W_up_v(value_states)
        value_states = self.v_proj(value_states)
        value_states = (
            value_states.transpose(0,1)
            .contiguous()
            .view(hidden_states.size(1),hidden_states.size(0),self.n_local_kv_heads,self.d_v)
        )
        return value_states


    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        from flash_attn import bert_padding
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        batch_size, q_length = sequence_mask.shape

        query_states = self.get_query_states(hidden_states, sequence_mask)
        key_states = self.get_key_states(hidden_states, sequence_mask)
        value_states = self.get_value_states(hidden_states, sequence_mask)

        store = self.get_local_store()  # In fact, collections.defaultdict?
        if store is not None:  # Inference case
            assert False, "Not implemented"

        else:  # Training case
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

        return {
            "hidden_states": output,
            "sequence_mask": sequence_mask,
        }


def custom_load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {
        id(module): f"{module_name}." for module_name, module in model.named_modules()
    }
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = (
        filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    )
    param_shard_metadata = {}
    missing_keys = []
    for name, param_or_buffer in tqdm(
        filtered_state_dict.items(),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Loading weights",
    ):
        # NOTE: extract how does the current model parameter are sharded
        # so that we can load optimizer checkpoints in this way
        param_shard_metadata[name] = {}
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[
                        sharded_info.global_ranks
                    ]
                    group_rank = dist.get_rank(group)

                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank),
                    parallel_context=parallel_context,
                )
                # TODO @nouamane: do we consider exp_size=1 expert_sharded?
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
            else:
                exp_tp_pp_rank_and_size = None
                is_expert_sharded = False

            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                prefix=param_root_folder,
                is_expert_sharded=is_expert_sharded,
            )

            if path.exists():
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    # TODO @thomasw21: Choose only a slice if we switch the TP topology
                    param_or_buffer[:] = fi.get_tensor("data")

            elif not path.parent.exists():
                missing_keys.append(name)
                # raise ValueError(
                #     f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                #     f"Couldn't find folder {path.parent} in checkpoint at {root_folder}"
                # )
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor
                # and load the specific shard we're interested in.
                if not param.is_sharded:
                    raise ValueError(
                        f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                    )
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(
                    path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors")
                )
                if len(shards_path) <= 0:
                    raise ValueError(
                        f"Could not find any shards {ObjectType.MODEL.value}_{suffix}*.safetensors in {path.parent}."
                        f"If you notice `.safetensors` in the middle of the name of some of the checkpoints files. You need to run `scripts/fix_checkpoint_bad_naming.py`."
                    )

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(
                        parallel_context, root_folder, param_save_path=shards_path[0]
                    )
                else:
                    current_checkpoint_version = None
                    try:
                        current_checkpoint_version = (
                            read_checkpoint_version_from_shard_file(
                                param_save_path=shards_path[0]
                            )
                        )
                    except CheckpointVersionFromShardFileException:
                        # The checkpoint version is read from the meta file
                        current_checkpoint_version = checkpoint_version
                    finally:
                        assert (
                            current_checkpoint_version == checkpoint_version
                        ), f"Checkpoint version mismatch at {shards_path[0]}."

                if checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer,
                        sharded_info=sharded_info,
                        shards_path=shards_path,
                        param_shard_metadata=param_shard_metadata[name],
                    )
                else:
                    raise ValueError(
                        f"Unsupported checkpoint version {checkpoint_version}"
                    )

        else:
            raise NotImplementedError(
                f"Parameters {param} should be a NanotronParameter"
            )
        
    low_rank = model.config.SVD["low_rank"] * model.config.num_key_value_heads
    logger.info(f"Low rank: {low_rank}")
    for missing_key in missing_keys:
        layer_idx = int(missing_key.split(".")[2])
        attn_module_prefix = f"model.decoder.{layer_idx}.pp_block.attn"
        attn_module = model.model.decoder[layer_idx].pp_block.attn
        base_name = f"model.decoder.{layer_idx}.pp_block.attn.qkv_proj.weight"
        path = get_path(
            base_name,
            type=ObjectType.MODEL,
            exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
            prefix=param_root_folder,
            is_expert_sharded=True,
        )
        with safe_open(path, framework="pt", device=str(param.device)) as fi:
            qkv_proj = fi.get_tensor("data").t()
        q_proj, k_proj, v_proj = qkv_proj.split(
            [
                attn_module.n_local_q_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_v,
            ],
            dim=-1,
        )
        if missing_key.endswith("W_down_k.weight"):
            dtype = filtered_state_dict[f"{attn_module_prefix}.W_down_k.weight"].dtype
            U,S,V = torch.svd(k_proj.to(dtype=torch.float32))
            U = U[:,:low_rank].to(dtype=dtype)
            S = S[:low_rank].to(dtype=dtype)
            V = V[:,:low_rank].to(dtype=dtype)
            filtered_state_dict[f"{attn_module_prefix}.W_down_k.weight"][:] = U.t()
            filtered_state_dict[f"{attn_module_prefix}.W_up_k.weight"][:] = (torch.diag(S) @ V.t()).t()
        elif missing_key.endswith("W_up_k.weight"):
            continue
        elif missing_key.endswith("W_down_v.weight"):
            dtype = filtered_state_dict[f"{attn_module_prefix}.W_down_v.weight"].dtype
            U,S,V = torch.svd(torch.eye(attn_module.d_model).to(dtype=torch.float32))
            U = U[:,:low_rank].to(dtype=dtype)
            S = S[:low_rank].to(dtype=dtype)
            V = V[:,:low_rank].to(dtype=dtype)
            filtered_state_dict[f"{attn_module_prefix}.W_down_v.weight"][:] = U.t()
            filtered_state_dict[f"{attn_module_prefix}.W_up_v.weight"][:] = (torch.diag(S) @ V.t()).t()
            # dtype = filtered_state_dict[f"{attn_module_prefix}.W_down_v.weight"].dtype
            # U,S,V = torch.svd(v_proj.to(dtype=torch.float32))
            # U = U[:,:low_rank].to(dtype=dtype)
            # S = S[:low_rank].to(dtype=dtype)
            # V = V[:,:low_rank].to(dtype=dtype)
            # filtered_state_dict[f"{attn_module_prefix}.W_down_v.weight"][:] = U.t()
            # filtered_state_dict[f"{attn_module_prefix}.W_up_v.weight"][:] = (torch.diag(S) @ V.t()).t()
        elif missing_key.endswith("W_up_v.weight"):
            continue
        elif missing_key.endswith("q_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.q_proj.weight"][:] = q_proj.t()
        elif missing_key.endswith("k_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.k_proj.weight"][:] = k_proj.t()
        elif missing_key.endswith("v_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.v_proj.weight"][:] = v_proj.t()
        else:
            raise ValueError(
                f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                f"Couldn't find folder {path.parent} in checkpoint at {root_folder}."
                f"Missing key: {missing_key}"
            )

    return param_shard_metadata


def low_rank_patch_nt(rope_cfg=None):
    llama.CausalSelfAttention = CustomCausalSelfAttention
    if not hasattr(nt_weights, "original_load_weights"):
        nt_weights.original_load_weights = nt_weights.load_weights
        nt_weights.load_weights = custom_load_weights
    nanotron.config.models_config.LlamaConfig = CustomLlamaConfig
    nanotron.trainer.CONFIG_TO_MODEL_CLASS.update(
        {"CustomLlamaConfig": nanotron.trainer.CONFIG_TO_MODEL_CLASS["LlamaConfig"]}
    )

    nanotron.trainer.load_weights = custom_load_weights
    from ..patch_func import create_custom_apply_rotary_pos_emb

    llama.LlamaRotaryEmbedding.apply_rotary_pos_emb = (
        create_custom_apply_rotary_pos_emb({"partial_rope_version":0})
    )
    from ..mla.NopeIndex import IndexForNope
    IndexForNope._qk_tensor_cache=torch.load(rope_cfg["qk_tensor_path"])
    IndexForNope._qk_tensor_path=rope_cfg["qk_tensor_path"]
