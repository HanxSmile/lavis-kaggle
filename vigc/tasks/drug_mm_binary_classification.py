from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np
from sklearn import metrics


@registry.register_task("drug_mm_binary_classification")
class DrugMMBinaryTrainEvalTask(BaseTask):

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

        response = model.generate(samples)
        pred, label, id_ = response["result"], response["label"], response["id"]

        for pred_, label_, uid_ in zip(pred, label, id_):
            this_res = {
                "pred": float(pred_),
                "label": int(label_),
                "id": str(uid_),
            }
            results.append(this_res)

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

        total_preds = [_["pred"] for _ in results]
        total_labels = [_["label"] for _ in results]

        total_preds = np.array(total_preds)
        total_labels = np.array(total_labels)

        metrics_result = {}

        fpr, tpr, thresholds = metrics.roc_curve(total_labels, total_preds)
        auc = metrics.auc(fpr, tpr)
        metrics_result["auc"] = auc

        log_stats = {split_name: metrics_result}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": metrics_result["auc"]}
        return res
