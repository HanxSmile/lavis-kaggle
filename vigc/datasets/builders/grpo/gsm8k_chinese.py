import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.grpo import Gsm8kChineseDataset

SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

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
        datasets["train"] = self.train_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "train"),
            system_prompt=SYSTEM_PROMPT
        )
        _ = datasets["train"][0]
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
        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.get("data_root"),
            split=cfg.get("split", "test"),
            system_prompt=SYSTEM_PROMPT
        )
        _ = datasets["eval"][0]
        return datasets
