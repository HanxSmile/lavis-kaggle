import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.grpo import Gsm8kChineseDataset


@registry.register_builder("gsm8k_chinese_train")
class Gsm8kChineseTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = Gsm8kChineseDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grpo/gsm8k_chinese_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Gsm8k Chinese train datasets ...")
        datasets = dict()

        cfg = self.config
        system_prompt = None
        system_prompt_name = cfg.get("system_prompt", None)
        if system_prompt_name is not None:
            system_prompt = registry.get_constant(system_prompt_name)
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "train"),
            system_prompt=system_prompt
        )
        _ = datasets["train"][0]
        logging.info(f"You are using system prompt: '{system_prompt}'")
        return datasets


@registry.register_builder("gsm8k_chinese_eval")
class Gsm8kChineseEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Gsm8kChineseDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grpo/gsm8k_chinese_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Gsm8k Chinese eval datasets ...")
        datasets = dict()

        cfg = self.config
        system_prompt = None
        system_prompt_name = cfg.get("system_prompt", None)
        if system_prompt_name is not None:
            system_prompt = registry.get_constant(system_prompt_name)
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "test"),
            system_prompt=system_prompt
        )
        logging.info(f"You are using system prompt: '{system_prompt}'")
        _ = datasets["eval"][0]
        return datasets
