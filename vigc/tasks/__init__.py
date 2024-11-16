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
from vigc.tasks.drug_mm_binary_classification import DrugMMBinaryTrainEvalTask

from vigc.tasks.whisper_asr_train_eval import WhisperASRTask
from vigc.tasks.translation_train_eval import TranslationTask
from vigc.tasks.translation_ds_train_eval import TranslationDeepSpeedTask
from vigc.tasks.tts_train_eval import TTSTask

from vigc.tasks.qwen_caption_task import QwenCaptionTask

from vigc.tasks.semantic_segmentation_task import SemanticSegmentationTask

from vigc.tasks.rag_embedding_task import RAGEmbeddingTask
from vigc.tasks.rag_rerank_task import RAGRerankTask


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
    "DrugMMBinaryTrainEvalTask",
    "WhisperASRTask",
]
