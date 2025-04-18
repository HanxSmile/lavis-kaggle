import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.sft.translation_json import TranslationJsonDataset
from vigc.datasets.datasets.sft.translation_json_test import TranslationJsonTestDataset


@registry.register_builder("translation_json_train")
class TranslationJsonTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = TranslationJsonDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/translation_json_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Translation Json SFT train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            source_lang=cfg.get("source_lang"),
            target_lang=cfg.get("target_lang"),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("translation_json_eval")
class TranslationJsonEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = TranslationJsonDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/translation_json_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Translation Json SFT eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            source_lang=cfg.get("source_lang"),
            target_lang=cfg.get("target_lang"),
        )
        _ = datasets["eval"][0]
        return datasets


@registry.register_builder("translation_json_test")
class TranslationJsonTestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = TranslationJsonTestDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/translation_json_test.yaml"
    }

    def build_datasets(self):
        logging.info("Building Translation Json SFT test datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
        )
        _ = datasets["eval"][0]
        return datasets
