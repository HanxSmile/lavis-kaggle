import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.drug_mm.balanced_multi_class_dataset import MultiClassDrugDataset

import warnings

warnings.filterwarnings("ignore")


@registry.register_builder("drug_mm_balanced_multi_class_train")
class DrugMMBalancedMultiClassTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultiClassDrugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/drug_mm/balanced_multi_class_train.yaml"
    }

    def build_datasets(self):
        self.build_processors()
        logging.info("Building DrugMM Balanced Multi Class train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            ann_path=build_info.ann_path,
            image_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            label_map=self.config.label_map
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("drug_mm_balanced_multi_class_eval")
class DrugMMBalancedMultiClassEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = MultiClassDrugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/drug_mm/balanced_multi_class_eval.yaml"
    }

    def build_datasets(self):
        self.build_processors()
        logging.info("Building DrugMM Balanced Multi Class eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            ann_path=build_info.ann_path,
            image_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            label_map=self.config.label_map,
            split="eval"
        )
        _ = datasets["eval"][0]
        return datasets
