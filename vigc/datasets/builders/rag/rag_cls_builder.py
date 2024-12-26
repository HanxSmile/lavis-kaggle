import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.rag.rag_cls_dataset import RAG_CLS_Dataset


@registry.register_builder("rag_cls_train")
class RAG_CLS_TrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = RAG_CLS_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rag/rag_cls_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building RAG CLS train datasets ...")
        datasets = dict()
        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
        )
        _ = datasets["train"][0]
        return datasets
