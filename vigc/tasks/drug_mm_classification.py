from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np
from sklearn import metrics


# reference: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification


@registry.register_task("drug_mm_classification_train_eval")
class DrugMMClassificationTrainEvalTask(BaseTask):

    def __init__(self, evaluate, label_map, report_metric=True):
        super().__init__()

        self.evaluate = evaluate
        self.label_map = label_map
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        label_map = run_cfg.get("label_map")

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            label_map=label_map
        )

    def valid_step(self, model, samples):
        results = []

        response = model.generate(samples)
        pred, label_index, id_ = response["result"], response["label_index"], response["id"]
        for pred_, label_index_, uid_ in zip(pred, label_index, id_):
            preds = [float(_) for _ in pred_]
            pred_index = preds.index(max(preds))
            this_res = {
                "pred": preds,
                "label_index": label_index_,
                "pred_index": pred_index,
                "id": int(uid_),
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

        total_preds = []
        total_labels = []
        month_preds = {}
        month_labels = {}

        for result in results:
            # result.keys: pred, label, pred_index, label_index
            pred, pred_index, label_index = result["pred"], result["pred_index"], result["label_index"]
            pred_class = self.label_map[pred_index]
            pred_score = pred[pred_index]

        total_preds = np.array(total_preds)
        total_labels = np.array(total_labels)

        month_preds = {k: np.array(v) for k, v in month_preds.items()}
        month_labels = {k: np.array(v) for k, v in month_labels.items()}

        month_preds["Total"] = total_preds
        month_labels["Total"] = total_labels

        metrics_result = {}
        for k in month_preds:
            preds = month_preds[k]
            labels = month_labels[k]
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            auc = metrics.auc(fpr, tpr)
            metrics_result[f"{k}_auc"] = auc

        log_stats = {split_name: metrics_result}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": metrics_result["Total_auc"]}
        return res
