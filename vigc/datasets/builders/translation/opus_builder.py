import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.translation.opus import OpusDataset


@registry.register_builder("opus_train")
class OpusTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpusDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/translation/opus_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Opus-100 Translation train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            source_key=cfg.get("source_key"),
            target_key=cfg.get("target_key"),
            split=cfg.get("split", None),
            switch_lang_flag=cfg.get("switch_lang_flag", False),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("opus_eval")
class OpusEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = OpusDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/translation/opus_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Opus-100 Translation eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            source_key=cfg.get("source_key"),
            target_key=cfg.get("target_key"),
            split="test",
            switch_lang_flag=False
        )
        _ = datasets["eval"][0]
        return datasets
