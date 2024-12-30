import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.text_rec.ctc_encode_dataset import SimpleDataSet


@registry.register_builder("ctc_encode_train")
class CTCTextRecTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = SimpleDataSet
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/text_rec/ctc_encode_train.yaml"
    }

    def build_datasets(self):
        logging.info("Building CTC Encode Text Recognition train datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["train"] = self.train_dataset_cls(
            media_dir=cfg.media_dir,
            ann_file=cfg.ann_file,
            processor=self.vis_processors["train"],
            delimiter=cfg.get("delimiter", "\t"),
            max_text_length=cfg.get("max_text_length", 25),
            character_dict_path=cfg.get("character_dict_path", None),
            use_space_char=cfg.get("use_space_char", False),
        )
        _ = datasets["train"][0]
        return datasets


@registry.register_builder("ctc_encode_eval")
class CTCTextRecEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = SimpleDataSet
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/text_rec/ctc_encode_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building CTC Encode Text Recognition Eval datasets ...")
        self.build_processors()
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            media_dir=cfg.media_dir,
            ann_file=cfg.ann_file,
            processor=self.vis_processors["eval"],
            delimiter=cfg.get("delimiter", "\t"),
            max_text_length=cfg.get("max_text_length", 25),
            character_dict_path=cfg.get("character_dict_path", None),
            use_space_char=cfg.get("use_space_char", False),
        )
        _ = datasets["eval"][0]
        return datasets
