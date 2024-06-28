import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.common_voice_complex_aug import CommonVoiceConcat, CommonVoiceSplit, \
    CommonVoiceSplitAndConcat
from transformers import WhisperProcessor
from .get_augmentation import get_augmentation


@registry.register_builder("whisper_common_voice_concat_train")
class CommonVoiceConcatTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonVoiceConcat
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/common_voice/concat_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Common Voice Concat ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        transform = get_augmentation(musan_dir)
        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            concat_nums=cfg.get("concat_nums"),
            split=cfg.get("split", "train"),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_common_voice_split_train")
class CommonVoiceSplitTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonVoiceSplit
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/common_voice/split_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Common Voice Split ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        transform = get_augmentation(musan_dir)
        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            split_nums=cfg.get("split_nums"),
            split=cfg.get("split", "train"),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_common_voice_split_and_concat_train")
class CommonVoiceSplitAndConcatTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonVoiceSplitAndConcat
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/common_voice/split_and_concat_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Common Voice Split&Concat ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        transform = get_augmentation(musan_dir)
        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            split_nums=cfg.get("split_nums"),
            concat_nums=cfg.get("concat_nums"),
            split=cfg.get("split", "train"),
        )
        _ = datasets["train"][0]
        return datasets
