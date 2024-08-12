import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.asr_data.myaudio import MyAudioTest
from transformers import WhisperProcessor


@registry.register_builder("whisper_myaudio_eval")
class MyAudioEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = MyAudioTest
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/myaudio/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building Whisper MyAudio ASR eval datasets ...")
        datasets = dict()
        data_root = self.config.data_root

        cfg = self.config
        processor = WhisperProcessor.from_pretrained(cfg.model_name, language=cfg.language, task="transcribe")
        datasets["eval"] = self.eval_dataset_cls(
            processor=processor,
            data_root=data_root,
            pre_normalize=cfg.get("pre_normalize", False),
            language=cfg.get("language", None),
        )
        _ = datasets["eval"][0]
        return datasets
