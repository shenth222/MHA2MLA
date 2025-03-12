import argparse
import os
from dataclasses import asdict
from pprint import pformat

from lighteval.parsers import parser_accelerate, parser_baseline, parser_nanotron, parser_utils_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector
import yaml


CACHE_DIR = os.getenv("HF_HOME")


# copied from lighteval.main
def cli_evaluate():  # noqa: C901
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    parser.add_argument(
        "--partial_rope_config",
        type=str,
        required=True,
        help="Path to the RoPE configuration file. This file should contain the partial-RoPE configuration.",
    )
    parser.add_argument(
        "--is_mla",
        action="store_true",
        help="Whether the model is a MLA model. If True, the model will be evaluated using the MLA architecture.",
    )

    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser("accelerate", help="use accelerate and transformers as backend for evaluation.")
    parser_accelerate(parser_a)

    args = parser.parse_args()

    # Monkey patching for the partial rope
    from monkey_patch import partial_rope_monkey_patch, mla_monkey_patch
    with open(args.partial_rope_config,"r") as f:
        cfg_RoPE = yaml.load(f, Loader=yaml.FullLoader)
        if "RoPE" in cfg_RoPE:
            cfg_RoPE=cfg_RoPE["RoPE"]
    if args.is_mla:
        mla_monkey_patch(cfg_RoPE)
    else:
        partial_rope_monkey_patch(cfg_RoPE)

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as main_accelerate

        main_accelerate(args)
    else:
        print("You did not provide any argument. Exiting")


if __name__ == "__main__":
    cli_evaluate()
