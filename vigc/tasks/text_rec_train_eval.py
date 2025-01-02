from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
from .text_rec_metrics import RecMetric


@registry.register_task("text_rec_task")
class TextRecTask(BaseTask):

    def __init__(self, evaluate, report_metric=True, is_filter=False, ignore_space=True):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.metric = RecMetric(
            is_filter=is_filter,
            ignore_space=ignore_space,
        )

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        metric_cfg = run_cfg.get("metric_cfg")

        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            is_filter=metric_cfg.get("is_filter", False),
            ignore_space=metric_cfg.get("ignore_space", True),
        )

    def valid_step(self, model, samples):
        results = []
        gts = samples["gt"]
        ids = samples["id"]
        image_paths = samples["image_path"]
        preds = model.generate(samples)
        for id_, pred, gt, image_path in zip(ids, preds, gts, image_paths):
            pred, score = pred
            results.append({
                "id": id_,
                "image": image_path,
                "pred": pred,
                "score": score,
                "gt": gt,
            })

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
        gts = [_["gt"] for _ in results]
        preds = [_["pred"] for _ in results]
        self.metric.reset()
        metrics = self.metric(preds, gts)
        accuracy, norm_edit_dis = metrics["acc"], metrics["norm_edit_dis"]

        log_stats = {split_name: metrics}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": norm_edit_dis, "norm_edit_dis": norm_edit_dis, "accuracy": accuracy}
        return res
