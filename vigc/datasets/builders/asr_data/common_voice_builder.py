import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.common_voice import CommonVoiceTrain, CommonVoiceTest
from transformers import WhisperProcessor
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)


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
        transform = Compose(
            [
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
                Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                OneOf(
                    [
                        AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0,
                                           noise_transform=PolarityInversion(), p=1.0),
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                    ] if musan_dir is not None else [
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0), ],
                    p=0.2,
                ),
            ]
        )

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.get("data_root"),
            transform=transform,
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
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
        )
        _ = datasets["eval"][0]
        return datasets
