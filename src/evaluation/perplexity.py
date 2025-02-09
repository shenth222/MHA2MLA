import argparse
import logging
import os
import time
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
from numba import jit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import (
    QuantizedCache,
    QuantizedCacheConfig,
    HQQQuantizedCache,
    QuantoQuantizedCache,
)

logger = logging.getLogger(__name__)


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts


def normalize(weights: Sequence[int | float]) -> np.ndarray:
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty(
        (n_samples,), dtype="long"
    )  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):
        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = (
            current_samples[max_error_index] % dataset_sizes[max_error_index]
        )

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        token_size (int): Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
    ) -> None:
        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn(
                "dataset_folders should be of type List[str] but str was provided. Converting to List[str]"
            )
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=token_size,
                    shuffle=True,
                )
            )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [
            len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets
        ]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        item = self.datatrove_datasets[dataset][dataset_sample]

        ###
        item["labels"] = item["input_ids"]
        item["attention_mask"] = [1 for _ in range(self.sequence_length)]
        return item
        ###

    def build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch,
            weights=self.dataset_weights,
            dataset_sizes=self.dataset_lengths,
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate(
            [dataset_sample_index for _ in range(num_epochs)]
        )
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]

        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):
        logger.info(f"> Total number of samples: {len(self)}")
        logger.info(f"> Total number of tokens: {len(self) * self.sequence_length}")

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(
            self.dataset_index, len(self.dataset_folders)
        )
        for index, sample_count in enumerate(dataset_sample_count):
            logger.info(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
            )


def plot(
    output_dir: str = "outputs",
    title: Optional[str] = None,
    perplexity_limit: Optional[float] = None,
    skip_first: int = 100,
):
    output_dir = Path(output_dir)

    fig, ax = plt.subplots()
    ax.set_xlabel("Input Sequence Length")

    for file in output_dir.glob("*.csv"):
        experiment = file.stem
        if "v2" in experiment:
            continue
        print(f"Loading {experiment}")
        df = pd.read_csv(file)
        df = df.groupby(["input_length"]).mean()
        X = df.index[skip_first:]
        Y = df["overall_ppl"][skip_first:]
        Y = np.log(Y)
        ax.plot(X, Y, "-", label=f"{experiment} perplexity")
    ax.set_ylabel("Perplexity (log), lower is better")
    if perplexity_limit:
        ax.set_ylim(top=min(ax.get_ylim()[1], perplexity_limit))
    ax.legend(loc=[1, 2, 7][0])  # upper right, upper left, center right

    ax.set_title(
        title.replace("\\n", "\n")
        if title
        else "Log perplexity as a function of input lengths"
    )
    fig.tight_layout()

    return fig


QUANTIZED_CACHE_CLASS = {
    "quanto": QuantoQuantizedCache,
    "HQQ": HQQQuantizedCache,
}


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    experiment: str,
    cache_implementation: str,
    output_dir: str = "outputs",
    data_column: str = "text",
    num_samples: Optional[int] = 1,
    num_tokens: Optional[int] = None,
    overwrite: bool = False,
    backend: str = "quanto",
    nbits: int = 4,
    batch_size: int = 16,  # 添加batch_size参数
    is_smollm1: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment}.csv"
    tokenizer.pad_token = tokenizer.eos_token

    if output_file.exists() and not overwrite:
        raise ValueError(
            f"The {output_file!r} output file already exists - if you really want to override it, then use `--overwrite`."
        )

    logs = defaultdict(list)
    loss_fn = CrossEntropyLoss(reduction="none")

    # 收集所有样本
    all_data = list(itertools.islice(dataset, num_samples))

    # 将数据分成多个batch
    num_batches = (len(all_data) + batch_size - 1) // batch_size

    for batch_start in range(0, len(all_data), batch_size):
        batch_end = min(batch_start + batch_size, len(all_data))
        current_batch_size = batch_end - batch_start
        if not is_smollm1:
            batch_data = [item[data_column] for item in all_data[batch_start:batch_end]]

            # 首先获取原始token长度（不带padding的）
            original_lengths = tokenizer(
                batch_data,
                return_tensors=None,  # 不转换为tensor，这样可以得到原始长度
                truncation=False,
                padding=False,
            )
            original_token_lengths = [len(ids) for ids in original_lengths["input_ids"]]

            # 检查原始长度是否满足要求
            assert all(
                length >= num_tokens for length in original_token_lengths
            ), f"All samples must have original length >= num_tokens ({num_tokens}). Got lengths: {original_token_lengths}"

            # 对当前batch进行编码
            encodings = tokenizer(
                batch_data,
                return_tensors="pt",
                padding=True,  # 添加padding
                truncation=True,  # 允许截断
                max_length=num_tokens
                * 2,  # 设置最大长度为num_tokens的两倍，确保有足够长度
            )

            # 只使用前num_tokens个token
            batch_input_ids = encodings.input_ids[:, :num_tokens]
        else:

            batch_input_ids = torch.stack(
                [item["input_ids"] for item in all_data[batch_start:batch_end]]
            )
        seq_len = num_tokens
        pbar = tqdm(
            range(0, seq_len - 1),
            desc=f"Processing batch {batch_start//batch_size + 1}/{num_batches}",
        )
        num_processed_tokens = 0

        if cache_implementation == "quantized":
            cache_config = QuantizedCacheConfig(
                backend=backend,
                nbits=nbits,
                residual_length=0,
                compute_dtype=model.dtype,
                device=model.device,
            )
            past_key_values = QUANTIZED_CACHE_CLASS[backend](cache_config)
        else:
            past_key_values = None

        for idx in pbar:
            start_t = time.time()
            input_ids = batch_input_ids[:, idx : idx + 1].to(
                model.device
            )  # [current_batch_size, 1]

            with torch.no_grad():
                outputs = model(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
                logits = outputs.logits.view(
                    -1, model.config.vocab_size
                )  # [current_batch_size, vocab_size]
                past_key_values = outputs.past_key_values

                # 获取下一个token作为标签
                labels = (
                    batch_input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                )  # [current_batch_size]
                neg_log_likelihood = loss_fn(logits, labels)  # [current_batch_size]
                perplexity = neg_log_likelihood.exp()  # [current_batch_size]

                # 计算当前batch的平均值
                batch_nll = neg_log_likelihood.mean().item()
                batch_ppl = perplexity.mean().item()

            pbar.set_description(
                f"Batch {batch_start//batch_size + 1}/{num_batches} - "
                f"nll: {batch_nll:>5.2f}, ppl: {batch_ppl:>8.2f}"
            )

            # 为当前batch中的每个样本存储数据
            for batch_idx in range(current_batch_size):
                global_sample_idx = batch_start + batch_idx + 1  # 全局样本索引
                logs["data_idx"].append(global_sample_idx)
                logs["input_length"].append(idx + 1)
                logs["nll"].append(neg_log_likelihood[batch_idx].item())
                logs["ppl"].append(perplexity[batch_idx].item())
                logs["overall_ppl"].append(
                    torch.tensor(logs["nll"]).mean().exp().item()
                )
                logs["cuda_vram_allocated"].append(
                    torch.cuda.memory_allocated(0)
                    / 1024
                    / 1024
                    / 1024
                    / current_batch_size
                )  # in GB
                logs["latency"].append(time.time() - start_t)

            # 每500个token保存一次
            if num_processed_tokens % 500 == 0:
                try:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                except KeyboardInterrupt as ex:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                    raise ex

            num_processed_tokens += 1
            if num_tokens and num_processed_tokens >= num_tokens:
                break

        # 每个batch处理完后清空GPU缓存
        if past_key_values is not None:
            del past_key_values
        torch.cuda.empty_cache()


def main_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log_perplexity_limit", type=float, default=5.0)
    # Perplexity starts a bit unstable, so we skip the start
    parser.add_argument("--skip_first", type=int, default=100)

    args = parser.parse_args()

    figure = plot(
        output_dir=args.output_dir,
        title=args.title,
        perplexity_limit=args.log_perplexity_limit,
        skip_first=args.skip_first,
    )

    # Add your own code here if you'd like to change the figure
    save_path = Path(args.output_dir) / "perplexity_plot.png"
    plt.savefig(save_path, dpi=600)
    print(f"plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    # How to call this experiment?
    parser.add_argument("--experiment", type=str, default="main")
    parser.add_argument("--cache_implementation", type=str, default="quantized")

    # Model args
    # parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--backend", type=str, default="quanto", choices=["quanto", "HQQ"]
    )
    parser.add_argument("--nbits", type=int, default=4)

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19-test")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=5000)
    parser.add_argument("--dtype", type=str, default="fp16")

    # Where to log
    parser.add_argument("--output_dir", type=str, default="./perplexity/outputs")
    parser.add_argument("--overwrite", action="store_true")

    # MLA
    parser.add_argument("--is_mla", action="store_true")
    parser.add_argument("--is_partial_rope", action="store_true")
    parser.add_argument("--cfg_RoPE", type=str, default=None)

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")
    torch.set_default_dtype(dtype)

    # Monkey Patching
    if args.is_mla:
        import json, os
        from ..mla.mla_patch_hf import mla_patch_hf

        with open(os.path.join(args.model_name_or_path, "config.json"), "r") as f:
            model_config = json.load(f)
        mla_patch_hf(model_config["RoPE"])
    if args.is_partial_rope:
        import yaml

        # Monkey patching for the partial rope
        with open(args.cfg_RoPE, "r") as f:
            cfg_RoPE = yaml.load(f, Loader=yaml.FullLoader)
            if "RoPE" in cfg_RoPE:
                cfg_RoPE = cfg_RoPE["RoPE"]
        from ..patch_func_hf import create_custom_apply_rotary_pos_emb_hf
        from transformers.models.llama import modeling_llama

        modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(
            cfg_RoPE
        )
        if cfg_RoPE["partial_rope_version"] == 4:
            from ..patch_func_hf import (
                custom_forward_LlamaAttention,
                custom_forward_LlamaFlashAttention2,
                custom_forward_LlamaSdpaAttention,
            )

            modeling_llama.LlamaAttention.forward = custom_forward_LlamaAttention
            modeling_llama.LlamaFlashAttention2.forward = (
                custom_forward_LlamaFlashAttention2
            )
            modeling_llama.LlamaSdpaAttention.forward = (
                custom_forward_LlamaSdpaAttention
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        # attn_implementation="eager",
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code)
    )

    # Set up the dataset
    if args.dataset_name == "smollm1_corpus":
        dataset = Nanoset(
            dataset_folders=[
                "~/local/smollm1_corpus/fineweb-edu-dedup/",
                "~/local/smollm1_corpus/cosmopedia-v2/",
                "~/local/smollm1_corpus/python-edu/",
                "~/local/smollm1_corpus/open-web-math/",
                "~/local/smollm1_corpus/stackoverflow/",
            ],
            sequence_length=args.num_tokens,
            token_size=2,
            train_split_num_samples=args.num_samples,
            dataset_weights=[
                0.7,
                0.15,
                0.08,
                0.06,
                0.01,
            ],
            random_seed=1234,
        )
    else:
        dataset = load_dataset(
            args.dataset_name, args.task, split=args.split, streaming=True
        )

    compute_perplexity(
        model,
        tokenizer,
        dataset,
        args.experiment,
        args.cache_implementation,
        output_dir=args.output_dir,
        data_column=args.data_column,
        num_samples=args.num_samples,
        num_tokens=args.num_tokens,
        overwrite=args.overwrite,
        backend=args.backend,
        nbits=args.nbits,
        is_smollm1=(args.dataset_name == "smollm1_corpus"),
    )


if __name__ == "__main__":
    # main_plot()
    main()
