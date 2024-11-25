import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.perplexity_permutation import PerplexityPermutationDataset


@registry.register_builder("perplexity_permutation")
class PerplexityPermutationBuilder(BaseDatasetBuilder):
    eval_dataset_cls = PerplexityPermutationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/kaggle/perplexity_permutation.yaml"
    }

    def build_datasets(self):
        logging.info("Building Perplexity Permutation datasets ...")
        datasets = dict()

        cfg = self.config
        datasets["eval"] = self.eval_dataset_cls(
            initial_text=cfg.initial_text,
            permutation_idx=[int(_) for _ in cfg.permutation_idx],
        )
        _ = datasets["eval"][0]
        return datasets
