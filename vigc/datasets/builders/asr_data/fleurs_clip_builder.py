import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.fleurs_clip import FleursClipDataset
from transformers import WhisperProcessor
from .get_augmentation import get_augmentation


@registry.register_builder("whisper_clip_fleurs_train")
class FleursClipTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = FleursClipDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/clip_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs Clip ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        cfg = self.config
        transform = get_augmentation(musan_dir) if cfg.get("augmentation", True) else None
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            ann_path=cfg.get("ann_path"),
            transform=transform,
            split=cfg.get("split", "train"),
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
            clip_duration=cfg.get("clip_duration", 3.0),
            using_space_sep=cfg.get("using_space_sep", True),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_clip_fleurs_eval")
class FleursClipEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = FleursClipDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/fleurs/clip_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Fleurs Clip ASR eval datasets ...")
        datasets = dict()
        data_root = self.config.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            ann_path=cfg.get("ann_path"),
            clip_duration=cfg.get("clip_duration", 3.0),
            split=cfg.get("split", "test"),
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
            using_space_sep=cfg.get("using_space_sep", True),
        )
        _ = datasets["eval"][0]
        return datasets
