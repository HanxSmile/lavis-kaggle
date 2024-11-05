import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.rag.rag_dataset import RAGDataset


@registry.register_builder("rag_train")
class RAGTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = RAGDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rag/rag_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building RAG train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            query_prompt=cfg.get("query_prompt", None),
            passage_prompt=cfg.get("passage_prompt", None),
            group_size=cfg.get("group_size", 4),
            processor=None  # TODO
        )
        _ = datasets["train"][0]
        return datasets
