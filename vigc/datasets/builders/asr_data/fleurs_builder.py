import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.fleurs import FleursTrain, FleursTest, FleursConcatTest
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

        cfg = self.config
        transform = get_augmentation(musan_dir) if cfg.get("augmentation", True) else None
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            split=cfg.get("split", "train"),
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_fleurs_eval")
class FleursEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = FleursTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs ASR eval datasets ...")
        datasets = dict()
        data_root = self.config.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            split=cfg.get("split", "test"),
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["eval"][0]
        return datasets


@registry.register_builder("whisper_fleurs_concat_eval")
class FleursConcatTestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = FleursConcatTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/concat_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs ASR Concat eval datasets ...")
        datasets = dict()
        cfg = self.config
        data_root = cfg.data_root
        language = cfg.language
        processor = [WhisperProcessor.from_pretrained(cfg.model_name, language=_, task="transcribe") for _ in language]
        split = cfg.split

        datasets["eval"] = self.eval_dataset_cls(
            data_root=data_root,
            processor=processor,
            language=language,
            pre_normalize=cfg.get("pre_normalize", False),
            split=split,
        )
        _ = datasets["eval"][0]
        return datasets
