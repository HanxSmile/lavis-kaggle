"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask

from vigc.tasks.hms_train_eval import HMSClassifyTrainEvalTask
from vigc.tasks.chatphone_train_eval import ChatPhoneTrainEvalTask
from vigc.tasks.drug_mm_classification import DrugMMClassificationTrainEvalTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "HMSClassifyTrainEvalTask",
    "ChatPhoneTrainEvalTask",
    "DrugMMClassificationTrainEvalTask",
]
