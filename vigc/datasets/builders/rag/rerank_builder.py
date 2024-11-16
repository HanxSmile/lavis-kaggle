import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.rag.rerank_dataset import RerankDataset, RerankEvalDataset


@registry.register_builder("rerank_train")
class RerankTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = RerankDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rag/rerank_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building RAG Rerank train datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            query_prompt=cfg.get("query_prompt", None),
            passage_prompt=cfg.get("passage_prompt", None),
            group_size=cfg.get("group_size", 4),
            processor=self.text_processors["train"],
            prompt=cfg.get("prompt", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("rerank_eval")
class RerankEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = RerankEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rag/rerank_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building RAG Rerank eval datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            query_prompt=cfg.get("query_prompt", None),
            passage_prompt=cfg.get("passage_prompt", None),
            processor=self.text_processors["eval"],
            prompt=cfg.get("prompt", None),
        )
        _ = datasets["eval"][0]
        return datasets
