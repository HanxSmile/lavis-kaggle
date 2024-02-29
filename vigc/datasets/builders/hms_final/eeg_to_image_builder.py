import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.hms_final.eeg_to_image_dataset import EEG2ImageDataset

import warnings

warnings.filterwarnings("ignore")


@registry.register_builder("eeg_to_image_train")
class EEG2ImageTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = EEG2ImageDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/eeg_to_image_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS EEG2Image inputs train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            random_seed=self.config.get("random_seed", 42),
            split="train",
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("eeg_to_image_eval")
class EEG2ImageEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = EEG2ImageDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/eeg_to_image_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS EEG2Image Inputs eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            random_seed=self.config.get("random_seed", 42),
            split="eval",
        )
        _ = datasets["eval"][0]
        return datasets
