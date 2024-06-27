import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.translation.text_pair import TextPairDataset


@registry.register_builder("text_pair_train")
class TextPairTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextPairDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/translation/text_pair_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Text Pair Translation train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            source_key=cfg.get("source_key"),
            target_key=cfg.get("target_key"),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("text_pair_eval")
class TextPairEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = TextPairDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/translation/text_pair_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Text Pair Translation eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            source_key=cfg.get("source_key"),
            target_key=cfg.get("target_key"),
        )
        _ = datasets["eval"][0]
        return datasets
