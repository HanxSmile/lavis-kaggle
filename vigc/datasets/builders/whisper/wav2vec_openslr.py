import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.whisper.wav2vec_openslr import OpenSLRConcatAugDataset, OpenSLRSegAugDataset, \
    OpenSLRConcatSegAugDataset
from transformers import Wav2Vec2Processor
from audiomentations import (
    Resample,
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)


def get_transform(musan_dir):
    trans = Compose(
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
    return trans


@registry.register_builder("wav2vec_openslr_seg_aug_asr")
class Wav2VecOpenSLRSegAugASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenSLRSegAugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/openslr_seg_aug_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec OpenSLR Segment Augmentation ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir")
        transform = get_transform(musan_dir)

        cfg = self.config
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)

        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.data_root,
            ann_file=cfg.ann_file,
            transform=transform,
            seg_nums=cfg.get("seg_nums", 3),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("wav2vec_openslr_concat_aug_asr")
class Wav2VecOpenSLRConcatAugASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenSLRConcatAugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/openslr_concat_aug_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec OpenSLR Concat Augmentation ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir")
        transform = get_transform(musan_dir)

        cfg = self.config
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.data_root,
            transform=transform,
            ann_file=cfg.ann_file,
            seg_nums=cfg.get("seg_nums", 2),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("wav2vec_openslr_concat_seg_aug_asr")
class Wav2VecOpenSLRConcatSegAugASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenSLRConcatSegAugDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wav2vec_bengali_asr/openslr_concat_seg_aug_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Wav2vec OpenSLR Concat Seg Augmentation ASR train datasets ...")
        datasets = dict()
        musan_dir = self.config.get("musan_dir")
        transform = get_transform(musan_dir)

        cfg = self.config
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            processor=processor,
            data_root=cfg.data_root,
            transform=transform,
            ann_file=cfg.ann_file,
            concat_seg_nums=cfg.get("concat_seg_nums", 2),
            split_seg_nums=cfg.get("split_seg_nums", 3)
        )
        _ = datasets["train"][0]
        return datasets
