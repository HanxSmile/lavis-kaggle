import logging
import numpy as np
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.image_regression.image_ctr import ImageCTRDataset
from albumentations import (
    HorizontalFlip, VerticalFlip,
    Transpose, HueSaturationValue, RandomResizedCrop, SmallestMaxSize,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")


def prepare_transforms(image_size=384):
    train_transforms = Compose([
        RandomResizedCrop(image_size, image_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)

    train_transform = lambda _: train_transforms(image=np.array(_))["image"]

    val_transforms = Compose([
        SmallestMaxSize(max_size=image_size),
        CenterCrop(image_size, image_size, p=1.),
        # Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    val_transform = lambda _: val_transforms(image=np.array(_))["image"]
    return train_transform, val_transform


@registry.register_builder("image_ctr_train")
class ImageCTRTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageCTRDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/image_regression/image_ctr_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Image CTR train datasets ...")
        build_info = self.config.build_info
        image_size = build_info.get("image_size", 384)
        image_processor = prepare_transforms(image_size)[0]
        datasets = dict()
        datasets["train"] = self.train_dataset_cls(
            ann_path=build_info.ann_path,
            media_path=build_info.media_path,
            image_processor=image_processor,
            split="train",
            fold_nums=build_info.get("fold_nums", 5),
            fold_idx=build_info.fold_idx,
            random_seed=build_info.get("random_seed", 42),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("image_ctr_eval")
class ImageCTREvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = ImageCTRDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/image_regression/image_ctr_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Image CTR eval datasets ...")
        build_info = self.config.build_info
        image_size = build_info.get("image_size", 384)
        image_processor = prepare_transforms(image_size)[1]
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            ann_path=build_info.ann_path,
            image_processor=image_processor,
            split="eval",
            media_path=build_info.media_path,
            fold_nums=build_info.get("fold_nums", 5),
            fold_idx=build_info.fold_idx,
            random_seed=build_info.get("random_seed", 42),
        )
        _ = datasets["eval"][0]
        return datasets
