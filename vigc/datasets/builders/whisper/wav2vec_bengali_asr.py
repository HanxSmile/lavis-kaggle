import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.whisper.bengali_wav2vec_asr import Wav2VecBengaliASR, Wav2VecBengaliASRTest
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


@registry.register_builder("wav2vec_bengali_asr")
class Wav2VecBengaliASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = Wav2VecBengaliASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec Bengali ASR train datasets ...")
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root
        musan_dir = self.config.get("musan_dir")
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
        processor = WhisperProcessor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=data_root,
            split="train",
            transform=transform,
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("wav2vec_bengali_asr_eval")
class Wav2VecBengaliASREvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Wav2VecBengaliASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec Bengali ASR eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name)

        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            split="valid",
        )
        _ = datasets["eval"][0]
        return datasets


@registry.register_builder("wav2vec_bengali_asr_test")
class Wav2VecBengaliASRTestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Wav2VecBengaliASRTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/test.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec Bengali ASR test datasets ...")
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name)
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
        )
        _ = datasets["eval"][0]
        return datasets
