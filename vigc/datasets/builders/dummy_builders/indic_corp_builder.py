import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dummy_datasets.indic_corp import IndicCorp


@registry.register_builder("indic_corp")
class BengaliIndicCorpEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = IndicCorp
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bengali_asr/indic_corp.yaml"
    }

    def build_datasets(self):
        logging.info("Building Bengali Indic Corp eval datasets ...")
        annotation = self.config.annotation
        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            anno_path=annotation
        )
        _ = datasets["eval"][0]
        return datasets
