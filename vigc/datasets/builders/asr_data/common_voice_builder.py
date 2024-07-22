import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.common_voice import CommonVoiceTrain, CommonVoiceTest
from transformers import WhisperProcessor
from .get_augmentation import get_augmentation


@registry.register_builder("whisper_common_voice_train")
class CommonVoiceTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonVoiceTrain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/common_voice/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper Common Voice ASR train datasets ...")
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
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_common_voice_eval")
class CommonVoiceEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = CommonVoiceTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/common_voice/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper CommonVoice ASR eval datasets ...")
        datasets = dict()
        data_root = self.config.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            split=cfg.get("split", "test"),
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["eval"][0]
        return datasets
