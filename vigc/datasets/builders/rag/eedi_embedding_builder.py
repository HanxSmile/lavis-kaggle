import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.rag.eedi_embedding_dataset import EediEvalDataset


@registry.register_builder("eedi_eval")
class EediEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = EediEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rag/eedi_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Eedi eval datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            misconception_root=cfg.get("misconception_root"),
            fold_idx=cfg.get("fold_idx"),
        )
        _ = datasets["eval"][0]
        return datasets
