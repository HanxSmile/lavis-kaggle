import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.instruct_blip.vqga_eval import A_OKVQA_VQGA_EvalDataset, COCO2017_Eval_Dataset, \
    OKVQA_VQGA_EvalDataset, VQAv2_VQGA_EvalDataset, LlavaEvalDataset, COCO2017_JiaHui_Eval_Dataset, \
    Object365_Eval_Dataset


@registry.register_builder("instruct_blip_aokvqa_vqga_eval")
class AOKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = A_OKVQA_VQGA_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/a-okvqa/vqga_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building A-OKVQA VQGA Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_jiahui_coco2017_vqga_test")
class COCO_Jiahui_VQGBuilder(BaseDatasetBuilder):
    eval_dataset_cls = COCO2017_JiaHui_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/test_jiahui_vqga.yaml"
    }

    def build_datasets(self):
        logging.info("Building COCO2017 JiaHui VQGA Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = self.config.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_coco2017_vqga_test")
class COCOPseudoEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = COCO2017_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/vqga_test.yaml"
    }

    def build_datasets(self):
        logging.info("Building COCO2017 VQGA Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        filter_dataset = self.config.get("filter", [])
        anno_path = build_info.annotation,
        image_id_path = build_info.image_ids
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
            image_ids_file=image_id_path,
            filter=filter_dataset
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_object365_vqga_test")
class COCOPseudoEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Object365_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/vqga_test_object365.yaml"
    }

    def build_datasets(self):
        logging.info("Building Object365 VQGA Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        filter_dataset = self.config.get("filter", [])
        anno_path = build_info.annotation,
        image_id_path = build_info.image_ids
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
            image_ids_file=image_id_path,
            filter=filter_dataset
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_okvqa_vqga_eval")
class OKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = OKVQA_VQGA_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/vqga_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building OKVQA VQGA Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_vqav2_vqga_eval")
class VQAv2EvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VQAv2_VQGA_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/vqga_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building VQAv2 VQGA Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_llava_vqga_eval")
class LlavaVQGAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = LlavaEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava_instruct150k/vqga_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building LLava VQGA Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets
