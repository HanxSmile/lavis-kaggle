import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.sft.tagengo_gpt4_dataset import TagengoGPT4Dataset


@registry.register_builder("tagengo_gpt4_sft_train")
class TagengoGPT4TrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = TagengoGPT4Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sft/tagengo_gpt4_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Tagengo GPT4 SFT train datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            language=cfg.get("language", None),
            use_system_prompt=cfg.get("use_system_prompt", False),
        )
        _ = datasets["train"][0]
        return datasets
