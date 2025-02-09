import torch
import logging
logger = logging.getLogger(__name__)

class SvdInit:
    @staticmethod
    def method_I(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = (U_k[:, :r] + U_v[:, :r]) / 2
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_II(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down_k = U_k
        W_down_v = U_v
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_III(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_IV(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_down = (W_down_k + W_down_v) / 2
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_V(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        W_down = U_k
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.matmul(W_down_pseudo_inv, v)
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VI(k, v, r=8):
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = U_v
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.matmul(W_down_pseudo_inv, k)
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VII(k, v, r=8):
        U_kv, S_kv, V_kv = torch.svd(torch.cat([k, v], dim=1))
        U_kv, S_kv, V_kv = U_kv[:, :r], S_kv[:r], V_kv[:, :r]
        W_down = U_kv
        split_sizes = [k.size(1), v.size(1)]
        W_up_k, W_up_v = torch.split(V_kv, split_sizes, dim=0)
        W_up_k = torch.diag(S_kv) @ W_up_k.t()
        W_up_v = torch.diag(S_kv) @ W_up_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def init(k, v, svd_method=1, r=8):
        assert k.dtype == v.dtype, "k and v must have the same dtype"
        logger.info(f"Using SVD method {svd_method} with rank {r}")
        original_dtype = k.dtype
        k=k.to(torch.float32)
        v=v.to(torch.float32)
        versions = {
            1: SvdInit.method_I,
            2: SvdInit.method_II,
            3: SvdInit.method_III,
            4: SvdInit.method_IV,
            5: SvdInit.method_V,
            6: SvdInit.method_VI,
            7: SvdInit.method_VII,
        }
        W_down_k, W_up_k, W_down_v, W_up_v = versions[svd_method](k, v, r)
        W_down_k = W_down_k.to(original_dtype)
        W_up_k = W_up_k.to(original_dtype)
        if W_down_v is not None:
            W_down_v = W_down_v.to(original_dtype)
        W_up_v = W_up_v.to(original_dtype)
        return W_down_k, W_up_k, W_down_v, W_up_v
