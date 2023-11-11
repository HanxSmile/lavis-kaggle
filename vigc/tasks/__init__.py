"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.tasks.caption_train_eval import InstructBlipCaptionTask
from vigc.tasks.llava_150k_gen import InstructBlipLLavaVQGATask
from vigc.tasks.vqa_train_eval import InstructBlipVQATask
from vigc.tasks.vqg_test import InstructBlipVQGTask
from vigc.tasks.bengali_asr import WhisperBengaliASRTask
from vigc.tasks.bengali_asr_infer import BengaliASRInferTask

from vigc.tasks.indic_corp_infer import BengaliIndicCorpInferTask
from vigc.tasks.ema_bengali_asr import EmaBengaliASRTask
from vigc.tasks.awp_bengali_asr import AwpBengaliASRTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "InstructBlipCaptionTask",
    "InstructBlipLLavaVQGATask",
    "InstructBlipVQATask",
    "InstructBlipVQGTask",
    "WhisperBengaliASRTask",
    "BengaliASRInferTask",
    "BengaliIndicCorpInferTask",
    "EmaBengaliASRTask",
    "AwpBengaliASRTask"
]
