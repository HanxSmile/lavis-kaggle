from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np


@registry.register_task("semantic_segmentation_task")
class SemanticSegmentationTask(BaseTask):

    def __init__(self, evaluate, report_metric=True, metric="iou"):
        super().__init__()
        assert metric in ["iou", "dice"]
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.metric = metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        metric = run_cfg.get("metric", "iou")
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            metric=metric,
        )

    def valid_step(self, model, samples):
        results = []
        names = samples["name"]
        preds = model.generate(samples)
        dice_scores = preds["dice_score"]
        iou_scores = preds["iou_score"]
        for name, dice_score, iou_score in zip(names, dice_scores, iou_scores):
            results.append({
                "name": name,
                "dice_score": dice_score.tolist(),
                "iou_score": iou_score.tolist(),
            })

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="name",
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
        dice_scores = [[_["dice_score"] for _ in results]]
        iou_scores = [_["iou_score"] for _ in results]
        dice_score = np.mean(dice_scores) * 100
        iou_score = np.mean(iou_scores) * 100

        log_stats = {split_name: {"dice": dice_score, "iou": iou_score}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        if self.metric == "iou":
            agg_metrics = iou_score
        else:
            agg_metrics = dice_score
        res = {"agg_metrics": agg_metrics, "iou": iou_score, "dice": dice_score}
        return res
