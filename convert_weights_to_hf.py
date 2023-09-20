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

PROCESSOR_NAME = {
    "bengali_wav2vec": "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
    "bengali_indicwav2vec": "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
    "bengali_indicwav2vec_shrink": "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
    "bengali_indicwav2vec_pro": "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
    "bengali_umong_wav2vec": "/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
}

MODEL_NAME = {
    "bengali_wav2vec": "/mnt/petrelfs/hanxiao/work/bengali_utils/cv_bn_bestModel_1",
    "bengali_indicwav2vec": "/mnt/lustre/hanxiao/work/bengali_utils/indic_wav2vec",
    "bengali_indicwav2vec_shrink": "/mnt/lustre/hanxiao/work/bengali_utils/shrink_indicwav2vec_v1_bengali",
    "bengali_indicwav2vec_pro": "/mnt/lustre/hanxiao/work/bengali_utils/pro_indicwav2vec_v1_bengali",
    "bengali_umong_wav2vec": "/mnt/lustre/hanxiao/work/bengali_utils/wav2vec2-large-mms-1b-bengali",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--src-path", required=True)
    parser.add_argument("--dst-path", required=True, help="path to save to hf weights.")

    args = parser.parse_args()

    return args


def main():
    cfg = parse_args()
    # from vigc.models.whisper.bengali_wav2vec import BengaliWav2Vec
    if cfg.model_type == "bengali_wav2vec":
        model = BengaliWav2Vec(
            model_name=MODEL_NAME[cfg.model_type],
            processor_name=PROCESSOR_NAME[cfg.model_type]
        )
    elif cfg.model_type == "bengali_umong_wav2vec":
        model = BengaliUmongWav2Vec(
            model_name=MODEL_NAME[cfg.model_type],
            processor_name=PROCESSOR_NAME[cfg.model_type]
        )
    elif cfg.model_type == "bengali_1b":
        model = Bengali1BWav2Vec(
            model_name="/mnt/petrelfs/hanxiao/work/lavis-kaggle/vigc/output/wav2vec_bengali_asr/facebook-1b-finetuned-stage-2/best_hf",
            processor_name="/mnt/petrelfs/hanxiao/work/bengali_utils/wav2vec2-xls-r-300m-bengali",
        )
    else:
        model = BengaliIndicWav2Vec(
            model_name=MODEL_NAME[cfg.model_type],
            processor_name=PROCESSOR_NAME[cfg.model_type]
        )
    model.load_checkpoint(cfg.src_path)
    hf_model = model.model
    hf_model.save_pretrained(cfg.dst_path)


if __name__ == "__main__":
    main()
