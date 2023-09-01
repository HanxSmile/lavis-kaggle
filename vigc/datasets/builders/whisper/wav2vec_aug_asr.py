import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.whisper.wav2vec_aug_asr import Wav2VecSegAugASR
from vigc.datasets.datasets.whisper.wav2vec_concat_asr import Wav2VecConcatAugASR
from transformers import Wav2Vec2Processor
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


@registry.register_builder("wav2vec_seg_aug_asr")
class Wav2VecSegAugASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = Wav2VecSegAugASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/seg_aug_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec Segment Augmentation ASR train datasets ...")
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
                        # AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0,
                        AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=3.0, max_snr_in_db=30.0,
                                           noise_transform=PolarityInversion(), p=1.0),
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                    ] if musan_dir is not None else [
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0), ],
                    p=0.5,
                ),
            ]
        )

        cfg = self.config
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=data_root,
            split="train",
            transform=transform,
            split_style=cfg.get("split_style", "default"),
            fold_idx=cfg.get("fold_idx", None),
            fold_nums=cfg.get("fold_nums", None),
            seed=cfg.get("seed", None),
            seg_nums=cfg.get("seg_nums", 3),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("wav2vec_concat_aug_asr")
class Wav2VecConcatAugASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = Wav2VecConcatAugASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/concat_aug_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec Concat Augmentation ASR train datasets ...")
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
                        AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=3.0, max_snr_in_db=30.0,
                                           noise_transform=PolarityInversion(), p=1.0),
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                    ] if musan_dir is not None else [
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0), ],
                    p=0.5,
                ),
            ]
        )

        cfg = self.config
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=data_root,
            split="train",
            transform=transform,
            split_style=cfg.get("split_style", "default"),
            fold_idx=cfg.get("fold_idx", None),
            fold_nums=cfg.get("fold_nums", None),
            seed=cfg.get("seed", None),
            seg_nums=cfg.get("seg_nums", 2),
        )
        _ = datasets["train"][0]
        return datasets
