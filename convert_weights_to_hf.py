"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import vigc.tasks as tasks
from vigc.common.config import Config
from vigc.common.dist_utils import get_rank, init_distributed_mode
from vigc.common.logger import setup_logger
from vigc.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from vigc.common.registry import registry
from vigc.common.utils import now

# imports modules for registration
from vigc.datasets.builders import *
from vigc.models import *
from vigc.processors import *
from vigc.runners import *
from vigc.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--src-path", required=True)
    parser.add_argument("--dst-path", required=True, help="path to save to hf weights.")
    parser.add_argument("--model-name", default="/mnt/petrelfs/hanxiao/work/bengali_utils/cv_bn_bestModel_1")
    parser.add_argument(
        "--processor-name",
        default="/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali")

    args = parser.parse_args()

    return args


def main():
    cfg = parse_args()
    # from vigc.models.whisper.bengali_wav2vec import BengaliWav2Vec
    model = BengaliWav2Vec(
        model_name=cfg.model_name,
        processor_name=cfg.processor_name
    )
    model.load_checkpoint(cfg.src_path)
    hf_model = model.model
    hf_model.save_pretrained(cfg.dst_path)


if __name__ == "__main__":
    main()
