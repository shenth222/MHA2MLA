import torch
import sys
sys.path.append("/data/shenth/work/MHA2MLA/src")
from HDM.utils import load_HDM_res, load_model
from llama_patch import enable_llama_patch
from transformers import AutoConfig, AutoTokenizer
import lm_eval
from lm_eval.utils import setup_logging, simple_parse_args_string
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)
from lm_patch import simple_evaluate
import json
from loguru import logger

def mean_merge(input_layers: list):
    tensor = sum(input_layers)
    return tensor / len(input_layers)

def merge_layers(input: list, rank=512, group_strategy=None, merge_imp="mean"):
    num_layers = len(input)
    if group_strategy is None:
        group_strategy = [list(range(num_layers))]
    merge_res = []
    for group_info in group_strategy:
        res = mean_merge(input[group_info[0]:group_info[-1]+1])
        merge_res.append(res)
    return merge_res

if __name__ == "__main__":
    # enable_patch()
    # config = AutoConfig.from_pretrained("/data/shenth/models/llama/2-7b-hf")
    # config._attn_implementation = "eager"
    # tokenizer = AutoTokenizer.from_pretrained("/data/shenth/models/llama/2-7b-hf")
    # model = load_model(config, "cuda")
    # w1, h1, w2, h2 = load_HDM_res("../HDM_res", mode="Q", rank=512)
    # model.rebuild_weight(32, w1, h1, w2, h2, "Q")
    # w1, h1, w2, h2 = load_HDM_res("../HDM_res", mode="K", rank=512)
    # model.rebuild_weight(32, w1, h1, w2, h2, "K")
    # torch.cuda.empty_cache()
    # from IPython import embed
    # embed()
    # prompt = ["Wow! Today is a sunny"]
    # inputs = tokenizer(prompt, return_tensors="pt")
    # res = model.generate(inputs.input_ids.cuda(), max_new_tokens=10, use_cache=True, num_beams=1, do_sample=False, temperature=None, top_p=None)
    # outputs = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(outputs)

    enable_llama_patch()
    # initialize logging
    setup_logging("INFO") # optional, but recommended; or you can set up logging yourself

    task_manager = lm_eval.tasks.TaskManager()

    # models.huggingface: 594
    # torch.cuda.memory._record_memory_history()
    results = simple_evaluate( # call simple_evaluate
        model = "hf",
        model_args = {"pretrained": "/data/shenth/models/llama/2-7b-hf",
                      "attn_implementation": "eager"
        },
        tasks = ["piqa"],
        batch_size = "64", # auto:4
        device = "cuda",
        cache_requests = True,
        task_manager = task_manager
    )
    # torch.cuda.memory._dump_snapshot(f"{model_name}-{label}-xkv.pickle")
    mem = torch.cuda.max_memory_allocated() /1024 /1024 /1024
    logger.info(f"Max memory: {mem}GB")

    if results is not None:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    
    # 23.686293601989746GB
    # QKVO 21.686293601989746GB
    # 49.24 UPG 4096->512 18.639540672302246GB
    # 49.51 UPG 11008->512 18.639540672302246GB