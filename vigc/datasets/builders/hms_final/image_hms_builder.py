import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.hms_final.image_hms_dataset import ImageHMSDataset
import albumentations as albu

import warnings

warnings.filterwarnings("ignore")


def get_train_transform():
    transform = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        # albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
    ])

    def func(img):
        return transform(image=img)['image']

    return func


@registry.register_builder("image_hms_train")
class ImageHMSTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageHMSDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/image_hms_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS Image inputs train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            random_seed=self.config.get("random_seed", 42),
            low_resource=self.config.get("low_resource", True),
            split="train",
            processor=get_train_transform()
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("image_hms_eval")
class ImageHMSEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = ImageHMSDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hms/image_hms_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building HMS Image Inputs eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            data_root=build_info.data_root,
            ann_path=build_info.ann_path,
            fold=self.config.fold,
            fold_idx=self.config.fold_idx,
            low_resource=self.config.get("low_resource", True),
            random_seed=self.config.get("random_seed", 42),
            split="eval",
        )
        _ = datasets["eval"][0]
        return datasets
