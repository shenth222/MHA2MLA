"""
This script use to log evaluation results to wandb.

python3 log_eval_results_to_wandb.py --eval-path /path/to/eval/results --wandb-project project_name --wandb-name run_name

The folder that contains the evaluation results should have the following structure:
- 5000:
    results_x.json # where x is the ligheval's evaluation number
- 10000:
    ...
...
"""
import argparse
import json
import os
import re
from pathlib import Path

import wandb


def run(current_path: Path):
    def compute_avg_acc_of_a_benchmark(data, benchmark_prefix):
        sum_acc, sum_acc_norm, sum_acc_stderr, sum_acc_norm_stderr, count = 0, 0, 0, 0, 0
        for key, values in data.items():
            if f"{benchmark_prefix}:" in key or f"{benchmark_prefix}_" in key:
                # sum_acc += values["acc"]
                sum_acc_norm += values["acc_norm"]
                # sum_acc_stderr += values["acc_stderr"]
                sum_acc_norm_stderr += values["acc_norm_stderr"]
                count += 1

        average_acc_norm = sum_acc_norm / count if count else 0
        return average_acc_norm

    def compute_avg_acc_of_all_tasks(data):
        sum_acc_norm, count = 0, 0
        for _, value in data.items():
            sum_acc_norm += value
            count += 1

        average_acc_norm = sum_acc_norm / count if count else 0
        return average_acc_norm

    def precess_path(path:str):
        if path.endswith("_hf"):
            path=path[:-3]
        match = re.search(r"(\d+)$", path)
        if not match:
            return 0
        number = match.group(1)
        return int(number)


    list_checkpoints = os.listdir(current_path)
    sorted_list_checkpoints = sorted(list_checkpoints, key=precess_path)

    for item in sorted_list_checkpoints:
        item_path = os.path.join(current_path, item)
        if os.path.isdir(item_path):
            json_files = [f for f in os.listdir(item_path) if f.endswith(".json")]
            if len(json_files) == 1:
                json_file_path = os.path.join(item_path, json_files[0])

                with open(json_file_path, "r") as file:
                    eval_data = json.load(file)
                    iteration_step = precess_path(item)
                    # consumed_train_samples = eval_data["config_general"]["config"]["general"]["consumed_train_samples"]

                    logging_results = {}
                    eval_data["results"] = {
                        (k if "|" not in k else k.split("|")[1]): v
                        for k, v in eval_data["results"].items()
                    }
                    for name, data in eval_data["results"].items():
                        if name.startswith("mmlu") or name.startswith("arc") or name.startswith("all"):
                            continue
                        if "acc_norm" in data:
                            logging_results[f"{name}_acc_norm"] = data["acc_norm"]
                        else:
                            logging_results[f"{name}_acc_norm"] = data["qem"]

                    logging_results["mmlu:average_acc_norm"] = compute_avg_acc_of_a_benchmark(eval_data["results"], "mmlu")
                    logging_results["arc:average_acc_norm"] = compute_avg_acc_of_a_benchmark(eval_data["results"], "arc")
                    logging_results["all:average_acc_norm"] = compute_avg_acc_of_all_tasks(logging_results)
                    print(logging_results)

                    wandb.log(
                        {
                            **logging_results,
                            "iteration_step": iteration_step,
                            # "consumed_train_samples": consumed_train_samples,
                        },
                        step=iteration_step,
                    )

            elif len(json_files) > 1:
                print(f"More than one JSON file found in {item_path}. Skipping.")
            else:
                print(f"No JSON file found in {item_path}.")

        print(f"Checkpoint {item} is done. /n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True, help="Path of the lighteval's evaluation results")
    parser.add_argument(
        "--wandb-project", type=str, help="Path of the lighteval's evaluation results", default="nanotron_evals"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        required=True,
        help="Path of the lighteval's evaluation results",
        default="sanity_evals",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_path = args.eval_path
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={"eval_path": eval_path},
    )

    run(eval_path)
