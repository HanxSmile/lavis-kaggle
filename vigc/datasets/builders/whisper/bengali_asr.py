import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.whisper.bengali_asr import BengaliASR
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


@registry.register_builder("whisper_bengali_asr")
class BengaliASRBuilder(BaseDatasetBuilder):
    train_dataset_cls = BengaliASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bengali_asr/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Bengali ASR train datasets ...")
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root

        cfg = self.config
        feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_name)
        tokenizer = WhisperTokenizer.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        datasets["train"] = self.train_dataset_cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            processor=processor,
            data_root=data_root,
            split="train"
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("whisper_bengali_asr_eval")
class BengaliASREvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = BengaliASR
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bengali_asr/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Bengali ASR eval datasets ...")
        build_info = self.config.build_info
        datasets = dict()
        data_root = build_info.data_root

        cfg = self.config
        feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_name)
        tokenizer = WhisperTokenizer.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language="bn", task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            processor=processor,
            data_root=data_root,
            split="valid"
        )
        _ = datasets["eval"][0]
        return datasets
