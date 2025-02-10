import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.viton.dresscode import DressCodeDataset


@registry.register_builder("dresscode_train")
class DressCodeTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = DressCodeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/viton/dresscode_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building DressCode train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            dataroot_path=cfg.dataroot_path,
            phase="train",
            order="paired",
            category=cfg.get("category", ("dresses", "upper_body", "lower_body")),
            size=cfg.get("size", (512, 384)),
            clip_vit_path=cfg.get("clip_vit_path", "openai/clip-vit-large-patch14"),
            offset=cfg.get("offset", None),
            cloth_background_whitening=cfg.get("cloth_background_whitening", False),
            cloth_mask_augmentation_ratio=cfg.get("cloth_mask_augmentation_ratio", 1.0),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("dresscode_eval")
class DressCodeEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = DressCodeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/viton/dresscode_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building DressCode Translation eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            dataroot_path=cfg.dataroot_path,
            phase="test",
            order=cfg.order,
            category=cfg.get("category", ("dresses", "upper_body", "lower_body")),
            size=cfg.get("size", (512, 384)),
            clip_vit_path=cfg.get("clip_vit_path", "openai/clip-vit-large-patch14"),
            offset=cfg.get("offset", None),
            cloth_background_whitening=cfg.get("cloth_background_whitening", False),
        )
        _ = datasets["eval"][0]
        return datasets
