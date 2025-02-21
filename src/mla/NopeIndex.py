import torch
from transformers import AutoModel, AutoTokenizer
import os
import copy
import logging
logger = logging.getLogger(__name__)


class IndexForNope:
    _qk_tensor_path = None
    _qk_tensor_cache = None

    @staticmethod
    def get_index_for_nope_v0(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v1(rope_cfg, **kwargs):
        keep_dim = rope_cfg["top_k_rope_dim"]
        head_dim = kwargs["head_dim"]
        if keep_dim <= 0:
            nope_mask = torch.ones((head_dim), dtype=torch.bool)
        elif keep_dim >= head_dim:
            nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        else:
            half = head_dim // 2
            nope_mask = torch.ones((half), dtype=torch.bool)
            nope_mask[:keep_dim] = False
            nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v2(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        indices_to_remove = torch.arange(
            rope_cfg["uniform_start_point"], head_dim, rope_cfg["uniform_step"]
        )
        nope_mask = torch.ones(head_dim, dtype=torch.bool)
        nope_mask[indices_to_remove] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope_v3(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_dim, last_k_dim = rope_cfg["top_k_rope_dim"], rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        assert top_k_dim + last_k_dim <= half
        nope_mask = torch.zeros((half), dtype=torch.bool)
        nope_mask[top_k_dim : half - last_k_dim] = True
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v4(rope_cfg, **kwargs):
        if IndexForNope._qk_tensor_cache is None or rope_cfg["qk_tensor_path"] != IndexForNope._qk_tensor_path:
            with open(rope_cfg["qk_tensor_path"], "rb") as fin:
                IndexForNope._qk_tensor_cache = torch.load(fin)  # [layer_num, k_head_num, head_dim//2]
                IndexForNope._qk_tensor_path = rope_cfg["qk_tensor_path"]
                assert len(IndexForNope._qk_tensor_cache.shape) == 3
        qk_tensor = IndexForNope._qk_tensor_cache
        layer_idx = kwargs["layer_idx"]
        top_k_dim = rope_cfg["top_k_rope_dim"]
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        nope_mask = torch.ones_like(qk_tensor[layer_idx], dtype=torch.bool)
        nope_mask.scatter_(1, topk_indices, False)
        nope_mask = torch.cat([nope_mask, nope_mask], dim=-1)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v5(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        last_k_rope_dim = rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        nope_mask = torch.ones((half), dtype=torch.bool)
        nope_mask[half-last_k_rope_dim:half] = False
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope(rope_cfg, **kwargs):
        logger.info(f"rope_cfg: {rope_cfg}")
        version = rope_cfg["partial_rope_version"]
        versions = {
            0: IndexForNope.get_index_for_nope_v0,
            1: IndexForNope.get_index_for_nope_v1,
            2: IndexForNope.get_index_for_nope_v2,
            3: IndexForNope.get_index_for_nope_v3,
            4: IndexForNope.get_index_for_nope_v4,
            5: IndexForNope.get_index_for_nope_v5,
        }
        index_func = versions[version]
        nope_mask = index_func(rope_cfg, **kwargs)
        nope_mask = nope_mask.to(dtype=torch.bool)
        if version == 4:
            nope_mask = nope_mask.reshape(-1)
        else:
            nope_mask = nope_mask.repeat(repeats=(kwargs["head_num"],))
        return nope_mask