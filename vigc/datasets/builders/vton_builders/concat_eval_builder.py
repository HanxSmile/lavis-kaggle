import logging

from vigc.common.registry import registry
from ..base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.vton_datasets.concat_dataset import VtonConcatTestDataset


@registry.register_builder("vton_concat_test")
class VtonConcatTestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VtonConcatTestDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vton/concat_eval.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets: Concat Test...")
        self.build_processors()
        datasets = dict()

        build_info = self.config.build_info
        dataset_root_info = build_info.dataset_root_info,
        orders = build_info.orders
        dataset_names = build_info.datasets
        size = tuple(list(build_info.size))

        clip_vit_path = build_info.clip_vit_path

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            dataset_root_info=dataset_root_info,
            datasets=dataset_names,
            orders=orders,
            size=size,
            clip_vit_path=clip_vit_path
        )
        _ = datasets['eval'][0]

        return datasets
