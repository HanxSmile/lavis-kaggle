import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.tts.tts_train import VitsTTSTrain
from vigc.datasets.datasets.tts.custum_tts_train import CustumVitsTTSTrain
from vigc.datasets.datasets.tts.tts_eval import VitsTTSEval
from vigc.datasets.datasets.tts.vits_feature_extractor import VitsFeatureExtractor
from transformers import AutoTokenizer


@registry.register_builder("vits_train")
class VitsTTSTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = VitsTTSTrain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vits/train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Vits TTS train datasets ...")
        datasets = dict()

        cfg = self.config
        build_cfg = cfg.build_info
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        processor = VitsFeatureExtractor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            data_root=build_cfg.data_root,
            processor=processor,
            tokenizer=tokenizer,
            audio_column_name=build_cfg.get("audio_column_name", "audio"),
            text_column_name=build_cfg.get("text_column_name", "text"),
            max_duration_in_seconds=build_cfg.get("max_duration_in_seconds", 20.0),
            min_duration_in_seconds=build_cfg.get("min_duration_in_seconds", 0.0),
            max_tokens_length=build_cfg.get("max_tokens_length", 450),
            do_lower_case=build_cfg.get("do_lower_case", False),
            speaker_id_column_name=build_cfg.get("speaker_id_column_name", "speaker_id"),
            filter_on_speaker_id=build_cfg.get("filter_on_speaker_id", None),
            do_normalize=build_cfg.get("do_normalize", False),
            num_workers=build_cfg.get("num_workers", 4),
            uroman_path=build_cfg.get("uroman_path", None),
            split=build_cfg.split,
            language=build_cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("vits_custum_train")
class VitsTTSCustumTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = CustumVitsTTSTrain
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vits/custum_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building Vits Custum TTS train datasets ...")
        datasets = dict()

        cfg = self.config
        build_cfg = cfg.build_info
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        processor = VitsFeatureExtractor.from_pretrained(cfg.model_name)
        datasets["train"] = self.train_dataset_cls(
            data_root=build_cfg.data_root,
            processor=processor,
            tokenizer=tokenizer,
            max_duration_in_seconds=build_cfg.get("max_duration_in_seconds", 20.0),
            min_duration_in_seconds=build_cfg.get("min_duration_in_seconds", 0.0),
            max_tokens_length=build_cfg.get("max_tokens_length", 450),
            do_lower_case=build_cfg.get("do_lower_case", False),
            filter_on_speaker_id=build_cfg.get("filter_on_speaker_id", None),
            do_normalize=build_cfg.get("do_normalize", False),
            uroman_path=build_cfg.get("uroman_path", None),
            language=build_cfg.get("language", None),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("vits_eval")
class VitsTTSEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VitsTTSEval
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vits/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Vits TTS eval datasets ...")
        datasets = dict()
        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        build_cfg = cfg.build_info

        datasets["eval"] = self.eval_dataset_cls(
            data_root=cfg.data_root,
            tokenizer=tokenizer,
            speaker_nums=cfg.get("speaker_nums", None),
            max_tokens_length=cfg.get("max_tokens_length", 450),
            do_lower_case=build_cfg.get("do_lower_case", False),
            uroman_path=build_cfg.get("uroman_path", None),
            language=build_cfg.get("language", None),
        )
        _ = datasets["eval"][0]
        return datasets
