import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.fleurs import FleursTrain, FluersTest
from transformers import WhisperProcessor
from .get_augmentation import get_augmentation


@registry.register_builder("whisper_fleurs_train")
class FleursTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = FleursTrain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        transform = get_augmentation(musan_dir)
        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            split=cfg.get("split", "train"),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_fleurs_eval")
class FleursEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = FluersTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs ASR eval datasets ...")
        datasets = dict()
        data_root = self.config.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            split=cfg.get("split", "test"),
        )
        _ = datasets["eval"][0]
        return datasets
