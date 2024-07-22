import logging

from vigc.common.registry import registry
from ..base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.vton_datasets.simple_vitonhd import VitonHDDataset


@registry.register_builder("viton_hd_train")
class VitonHDTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = VitonHDDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vton/viton_hd_train.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets: VitonHD Train...")
        self.build_processors()

        build_info = self.config.build_info
        data_root = build_info.data_root,
        size = tuple(list(build_info.size))
        clip_vit_path = build_info.clip_vit_path

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            dataroot_path=data_root,
            phase="train",
            order="paired",
            size=size,
            clip_vit_path=clip_vit_path,
        )
        _ = datasets['train'][0]

        return datasets


@registry.register_builder("viton_hd_test")
class VitonHDTestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VitonHDDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vton/viton_hd_eval.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets: VitonHD Test...")
        self.build_processors()

        build_info = self.config.build_info
        data_root = build_info.data_root,
        order = build_info.order
        size = tuple(list(build_info.size))
        datasets = dict()
        clip_vit_path = build_info.clip_vit_path

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            dataroot_path=data_root,
            phase="test",
            order=order,
            size=size,
            clip_vit_path=clip_vit_path,
        )
        _ = datasets['eval'][0]

        return datasets
