"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.datasets.builders.base_dataset_builder import load_dataset_config

from vigc.common.registry import registry

from vigc.datasets.builders.instruct_blip.caption_builder import CCSBUBuilder
from vigc.datasets.builders.instruct_blip.vqa_eval_builder import (
    AOKVQAEvalBuilder,
    VQAv2EvalBuilder,
    OKVQAEvalBuilder,
)

from vigc.datasets.builders.instruct_blip.vqga_builder import (
    LlavaCompVQGABuilder,
    LlavaDescVQGABuilder,
    LlavaConvVQGABuilder,
    A_OKVQA_VQGABuilder,
    OKVQA_VQGABuilder,
    VQAv2_VQGABuilder,
    COCO_Pseudo_VQGABuilder,
    LlavaCompVQABuilder,
    LlavaDescVQABuilder,
    LlavaConvVQABuilder,
    A_OKVQA_VQABuilder,
    OKVQA_VQABuilder,
    VQAv2_VQABuilder,
    COCO_Pseudo_VQABuilder,
    LlavaCompVQGBuilder,
    LlavaDescVQGBuilder,
    LlavaConvVQGBuilder,
    A_OKVQA_VQGBuilder,
    OKVQA_VQGBuilder,
    COCO_Pseudo_VQGBuilder,
)
from vigc.datasets.builders.instruct_blip.vqga_eval_builder import (
    AOKVQAEvalBuilder,
    COCO_Jiahui_VQGBuilder,
    COCOPseudoEvalBuilder,
    OKVQAEvalBuilder,
    VQAv2EvalBuilder,
    LlavaVQGAEvalBuilder,
)

from vigc.datasets.builders.whisper.bengali_asr import (
    BengaliASRBuilder,
    BengaliASREvalBuilder,
    BengaliASRTestBuilder
)

from vigc.datasets.builders.whisper.wav2vec_bengali_asr import (
    Wav2VecBengaliASRBuilder,
    Wav2VecBengaliASREvalBuilder,
    Wav2VecBengaliASRTest
)

from vigc.datasets.builders.whisper.wav2vec_aug_asr import (
    Wav2VecSegAugASRBuilder,
    Wav2VecConcatAugASRBuilder
)

from vigc.datasets.builders.dummy_builders.indic_corp_builder import (
    BengaliIndicCorpEvalBuilder,
)

from vigc.datasets.builders.whisper.wav2vec_openslr import (
    Wav2VecOpenSLRSegAugASRBuilder,
    Wav2VecOpenSLRConcatAugASRBuilder,
    Wav2VecOpenSLRConcatSegAugASRBuilder
)

__all__ = [
    # "AOKVQA_Train_Builder",
    "VQAv2EvalBuilder",
    "OKVQAEvalBuilder",
    "AOKVQAEvalBuilder",
    "LlavaCompVQGABuilder",
    "LlavaDescVQGABuilder",
    "LlavaConvVQGABuilder",
    "A_OKVQA_VQGABuilder",
    "OKVQA_VQGABuilder",
    "VQAv2_VQGABuilder",
    "COCO_Pseudo_VQGABuilder",
    "LlavaCompVQABuilder",
    "LlavaDescVQABuilder",
    "LlavaConvVQABuilder",
    "A_OKVQA_VQABuilder",
    "OKVQA_VQABuilder",
    "VQAv2_VQABuilder",
    "COCO_Pseudo_VQABuilder",
    "LlavaCompVQGBuilder",
    "LlavaDescVQGBuilder",
    "LlavaConvVQGBuilder",
    "A_OKVQA_VQGBuilder",
    "OKVQA_VQGBuilder",
    "COCO_Pseudo_VQGBuilder",
    "AOKVQAEvalBuilder",
    "COCO_Jiahui_VQGBuilder",
    "COCOPseudoEvalBuilder",
    "OKVQAEvalBuilder",
    "VQAv2EvalBuilder",
    "LlavaVQGAEvalBuilder",
    "CCSBUBuilder",

    "BengaliASRBuilder",
    "BengaliASREvalBuilder",
    "BengaliASRTestBuilder",

    "Wav2VecBengaliASRBuilder",
    "Wav2VecBengaliASREvalBuilder",
    "Wav2VecBengaliASRTest",

    "Wav2VecSegAugASRBuilder",
    "Wav2VecConcatAugASRBuilder",

    "BengaliIndicCorpEvalBuilder",

    "Wav2VecOpenSLRSegAugASRBuilder",
    "Wav2VecOpenSLRConcatAugASRBuilder",
    "Wav2VecOpenSLRConcatSegAugASRBuilder"
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
                data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
