import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.greek_single_speaker_speech import GreekSingleSpeakerSpeechTrain
from transformers import WhisperProcessor
from .get_augmentation import get_augmentation


@registry.register_builder("whisper_greek_single_speaker_speech_train")
class GreekSingleSpeakerSpeechTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = GreekSingleSpeakerSpeechTrain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/greek_single_speaker_speech/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper GreekSingleSpeakerSpeech ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir", None)

        transform = get_augmentation(musan_dir)
        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets
