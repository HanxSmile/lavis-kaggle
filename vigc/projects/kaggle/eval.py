"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import os

import numpy as np
import torch
import json
import torch.backends.cudnn as cudnn
import vigc.tasks as tasks

from vigc.common.dist_utils import is_main_process
from vigc.common.config import Config
from vigc.common.registry import registry
from vigc.common.dist_utils import get_rank, init_distributed_mode
from vigc.common.logger import setup_logger
from vigc.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from vigc.common.utils import now

# imports modules for registration
from vigc.datasets.builders import *
from vigc.models import *
from vigc.processors import *
# from vigc.runners.runner_base import RunnerBase
from vigc.runners.runner_iter import RunnerIter
from vigc.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--dst-path", required=True, help="path to output directory.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()

    initial_text = "wreath hohoho merry from to as you have in the season of wonder workshop not that we believe it with joy hope peace chocolate candy peppermint fruitcake poinsettia candle star angel greeting card wrapping paper bow toy doll game puzzle cookie milk and eggnog snowglobe fireplace wish dream night kaggle"
    initial_permutation_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_logger()

    if not hasattr(cfg.run_cfg, "max_iters"):
        cfg.run_cfg.max_iters = 1
    if not hasattr(cfg.run_cfg, "iters_per_inner_epoch"):
        cfg.run_cfg.iters_per_inner_epoch = 1

    setup_seeds(cfg)

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    for i in range(len(initial_text.split())):
        this_permutation_ids = [(_ + i) % len(initial_text.split()) for _ in initial_permutation_ids]

        for dataset_name in cfg.datasets_cfg:
            cfg.datasets_cfg[dataset_name].initial_text = initial_text
            cfg.datasets_cfg[dataset_name].permutation_idx = this_permutation_ids
        datasets = task.build_datasets(cfg)

        cfg.pretty_print()

        runner = RunnerIter(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )
        runner.evaluate(skip_reload=True)

        if is_main_process():
            with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt")) as f:
                lines = f.readlines()
            line = [_.strip() for _ in lines if _.strip()][-1]
            result_info = json.loads(line)
            result_text, result_score = result_info["eval"]["text"], result_info["eval"]["score"]
            new_line = json.dumps({"text": result_text, "score": result_score, "permutation_idx": this_permutation_ids})
            with open(args.dst_path, "a") as f:
                f.write(new_line + "\n")
            initial_text = result_text


if __name__ == "__main__":
    main()
