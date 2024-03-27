import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.drug_mm.binary_class_dataset import BinaryClassDrugDataset

import warnings

warnings.filterwarnings("ignore")


@registry.register_builder("drug_mm_binary_class_train")
class DrugMMBinaryClassTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = BinaryClassDrugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/drug_mm/binary_class_train.yaml"
    }

    def build_datasets(self):
        self.build_processors()
        logging.info("Building DrugMM Binary Class train datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["train"] = self.train_dataset_cls(
            ann_path=build_info.ann_path,
            image_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            split="train"
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("drug_mm_binary_class_eval")
class DrugMMBinaryClassEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = BinaryClassDrugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/drug_mm/binary_class_eval.yaml"
    }

    def build_datasets(self):
        self.build_processors()
        logging.info("Building DrugMM Binary Class eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            ann_path=build_info.ann_path,
            image_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            split="eval"
        )
        _ = datasets["eval"][0]
        return datasets
