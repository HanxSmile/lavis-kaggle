import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.semantic_segmentation.cityscapes import CityScapes


@registry.register_builder("cityscapes_train")
class CityScapesTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CityScapes
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/semantic_seg/cityscapes_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building CityScapes train datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            ann_path=cfg.get("ann_path"),
            media_dir=cfg.get("media_dir"),
            processor=self.vis_processors["train"]
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("cityscapes_eval")
class CityScapesEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = CityScapes
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/semantic_seg/cityscapes_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building CityScapes eval datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            ann_path=cfg.get("ann_path"),
            media_dir=cfg.get("media_dir"),
            processor=self.vis_processors["eval"]
        )
        _ = datasets["eval"][0]
        return datasets
