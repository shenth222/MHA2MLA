import lm_eval
from lm_eval.utils import setup_logging, simple_parse_args_string
from lm_eval.loggers import EvaluationTracker
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import torch
import argparse
from argparse import Namespace
import math
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
import plotly.express as px
import random
from IPython import embed
import json
from lm_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)
import numpy as np
import random
import sys
sys.path.append("../../../../xKV/")
from utils import load_model_and_tokenizer, apply_kv_compress_patch
from loguru import logger
from llama_patch import enable_llama_xKV_patch_forward




def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark for cka")
    model_choices = ["135m", "2-7b", "360m", "1b", "3-8b"]
    parser.add_argument("--model_name", type=str, required=True, choices=model_choices, help="Model name for test")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for test")
    parser.add_argument("--load_path", type=str, help="Path to load qk states")
    parser.add_argument("--save_path", type=str, help="Path to save qk states")
    parser.add_argument("--figure_path", type=str, default="./figure/benchmark/", help="Path to save figure")
    rope_choices = ["high", "low", "uniform", "2-norm", "accumulate", "high-low", "full-rope", "contribution"]
    parser.add_argument("--rope_method", type=str, default="norm", choices=rope_choices, help="Partial RoPE method")
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser.add_argument("--rank_k", type=int, default=256, help="Rank for SVD compression of keys")
    parser.add_argument("--rank_v", type=int, default=768, help="Rank for SVD compression of values")
    parser.add_argument(
        '--layer_group_size',
        type=int,
        default=1,
        help='The number of layers that will be grouped and decompose jointly'
    )
    
    parser.add_argument(
        '--layer_merge_impl', 
        type=str, 
        default='svd',
        help='The implementation for layer merge'
    )
    parser.add_argument(
        '--slerp_t',
        type=float,
        default=0.5,
        help='The interpolation ratio for SLERP'
    )
    parser.add_argument(
        '--slerp_gamma',
        type=float,
        default=0.05,
        help='The gamma for identifying divergent token in SLERP',
    )
    
    # Merge control
    parser.add_argument("--merge_key", action="store_true", help="Enable merging for keys")
    parser.add_argument("--merge_value", action="store_true", help="Enable merging for values")
    parser.add_argument("--start_layer_idx", type=int, default=0, help="The starting layer index for layer merging")
    parser.add_argument("--end_layer_idx", type=int, default=-1, help="The ending layer index for layer merging. If -1, it will be the last layer.")
    parser.add_argument('--customized_merge_config', type=str, help='custom config file')
    return parser

MODEL_MAP = {
    "135m": "/data/shenth/models/SmolLM/135m",
    "360m": "/data/shenth/models/SmolLM/360m",
    "2-7b": "/data/shenth/models/llama/2-7b-hf",
    "1b": "/data/shenth/models/SmolLM/1b",
    "3-8b": "/data/shenth/models/llama/3.1-8b-ins"
}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task
    bsz = args.bsz
    load_path = args.load_path
    save_path = args.save_path
    figure_path = args.figure_path
    model_path = MODEL_MAP[model_name]
    rope_method = args.rope_method
    label = 'disable'

    from transformers import AutoTokenizer, LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(model_path)
    # exit()

    # model, tokenizer = load_model_and_tokenizer(model_path, use_flash_attn2=args.flash2)

    if args.customized_merge_config is not None:
        logger.info(f"Enable xKV with config {args.customized_merge_config}")
        enable_llama_xKV_patch_forward(args.customized_merge_config)
        label = 'enable'

    # initialize logging
    setup_logging("INFO") # optional, but recommended; or you can set up logging yourself

    task_manager = lm_eval.tasks.TaskManager()

    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    # models.huggingface: 594
    # torch.cuda.memory._record_memory_history()
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model = "hf",
        model_args = f"pretrained={model_path}",
        tasks = [task],
        batch_size = "64", # auto:4
        device = "cuda",
        cache_requests = True,
        evaluation_tracker = evaluation_tracker,
        task_manager = task_manager
    )
    # torch.cuda.memory._dump_snapshot(f"{model_name}-{label}-xkv.pickle")

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))