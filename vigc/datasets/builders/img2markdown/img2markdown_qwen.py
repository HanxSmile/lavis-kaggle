import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.img2markdown import QwenIm2MkdownDataset


@registry.register_builder("qwen_img2markdown_train")
class QwenIm2MkdownTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = QwenIm2MkdownDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/img2markdown/qwen_train.yaml"
    }
    LOG_INFO = "Image 2 Markdown Recognition Train For Qwen"

    def build_datasets(self):
        logging.info(f"Building {self.LOG_INFO} datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.ann_path
        meidia_dir = build_info.media_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            processor=self.vis_processors["train"],
            ann_path=anno_path,
            media_dir=meidia_dir,
        )
        print(datasets['train'][0])

        return datasets


@registry.register_builder("qwen_img2markdown_eval")
class QwenIm2MkdownEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = QwenIm2MkdownDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/img2markdown/qwen_eval.yaml"
    }
    LOG_INFO = "Image 2 Markdown Recognition Eval For Qwen"

    def build_datasets(self):
        logging.info(f"Building {self.LOG_INFO} datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.ann_path
        meidia_dir = build_info.media_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            processor=self.vis_processors["eval"],
            ann_path=anno_path,
            media_dir=meidia_dir,
        )
        print(datasets['eval'][0])

        return datasets
