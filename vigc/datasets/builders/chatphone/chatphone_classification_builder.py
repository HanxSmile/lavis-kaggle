import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.chatphone.chatphone_classification import ChatPhoneClassificationDataset

import warnings

warnings.filterwarnings("ignore")


@registry.register_builder("chatphone_train")
class ChatPhoneTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = ChatPhoneClassificationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/chatphone/chatphone_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building ChatPhone train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            ann_path=build_info.ann_path,
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("chatphone_eval")
class ChatPhoneEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = ChatPhoneClassificationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/chatphone/chatphone_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building ChatPhone eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            ann_path=build_info.ann_path,
        )
        _ = datasets["eval"][0]
        return datasets
