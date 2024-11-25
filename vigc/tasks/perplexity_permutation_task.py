from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np


@registry.register_task("perplexity_permutation")
class PerplexityPermutationEvalTask(BaseTask):

    def __init__(self, evaluate, report_metric=True):
        super().__init__()

        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        scores = model.generate(samples["text"])
        ids, texts = samples["id"], samples["text"]

        for id_, text, score in zip(ids, texts, scores):
            results.append({"id": id_, "text": text, "score": score})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        with open(eval_result_file) as f:
            results = json.load(f)

        all_scores = [_["score"] for _ in results]
        all_texts = [_["text"] for _ in results]

        min_idx = np.argmin(all_scores)
        min_score = all_scores[min_idx]
        best_text = all_texts[min_idx]

        min_idxs = np.argsort(all_scores)
        min_scores = [all_scores[_] for _ in min_idxs]
        min_texts = [all_texts[_] for _ in min_idxs]

        store_results = {"text": min_texts, "score": min_scores}

        with open(os.path.join(registry.get_path("result_dir"), "filter_results.json"), "w") as f:
            json.dump(store_results, f)

        metrics_result = {"score": min_score, "text": best_text}

        log_stats = {split_name: metrics_result}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": 10000 - metrics_result["score"]}
        return res
