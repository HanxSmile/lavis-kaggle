import logging
import cv2
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.hms.double_inputs import DoubleInputsDataset
import albumentations.augmentations.geometric.functional as F

import warnings

warnings.filterwarnings("ignore")


def get_train_transform():
    def func(img):
        return F.resize(img, height=512, width=512, interpolation=cv2.INTER_LINEAR)

    return func


def get_eval_transform():
    def func(img):
        return F.resize(img, height=512, width=512, interpolation=cv2.INTER_LINEAR)

    return func


@registry.register_builder("hms_double_inputs_train")
class HMSDoubleInputsTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = DoubleInputsDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/double_inputs_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS double inputs train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            random_seed=self.config.get("random_seed", 42),
            split="train",
            processor=get_train_transform()
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("hms_double_inputs_eval")
class HMSDoubleInputsEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = DoubleInputsDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/double_inputs_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS Double Inputs eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            random_seed=self.config.get("random_seed", 42),
            split="eval",
            processor=get_eval_transform()
        )
        _ = datasets["eval"][0]
        return datasets
