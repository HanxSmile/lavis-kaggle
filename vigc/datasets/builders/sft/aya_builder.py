import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.sft.aya_dataset import AyaInstructionDataset


@registry.register_builder("aya_sft_train")
class AyaSFTTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = AyaInstructionDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/aya_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Aya SFT train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "train"),
            language=cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("aya_sft_eval")
class AyaSFTEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = AyaInstructionDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/aya_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Aya SFT eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "test"),
            language=cfg.get("language", None),
        )
        _ = datasets["eval"][0]
        return datasets
