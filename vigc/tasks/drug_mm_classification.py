from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json


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
        pred, label, id_ = response["result"], response["label"], response["id"]

        for pred_, label_, uid_ in zip(pred, label, id_):
            preds = [float(_) for _ in pred_]
            label_ = [int(_) for _ in label_]
            pred_index = preds.index(max(preds))
            label_index_ = label_.index(max(label_))
            this_res = {
                "pred": preds,
                "label_index": label_index_,
                "pred_index": pred_index,
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
        """
            this_res = {
                "pred": preds,
                "label_index": label_index_,
                "pred_index": pred_index,
                "id": str(uid_),
            }
        """

        with open(eval_result_file) as f:
            results = json.load(f)

        total_precisions = {}
        total_recalls = {}

        for result in results:
            pred, label_index, pred_index = result["pred"], result["label_index"], result["pred_index"]
            gt_label_name = self.label_map[label_index]
            pred_label_name = self.label_map[pred_index]
            hit = int(gt_label_name == pred_label_name)
            if pred_label_name not in total_precisions:
                total_precisions[pred_label_name] = {"np": 1, "tp": hit}
            else:
                total_precisions[pred_label_name]["np"] += 1
                total_precisions[pred_label_name]["tp"] += hit
            if gt_label_name not in total_recalls:
                total_recalls[gt_label_name] = {"p": 1, "hit": hit}
            else:
                total_recalls[gt_label_name]["p"] += 1
                total_recalls[gt_label_name]["hit"] += hit

        total_recall = {f"{k}_recall": v["hit"] / v["p"] for k, v in total_recalls.items()}
        total_recall["total_recall"] = sum(total_recall.values()) / len(total_recall)
        total_precision = {f"{k}_precision": v["tp"] / v["np"] for k, v in total_precisions.items()}
        total_precision["total_precision"] = sum(total_precision.values()) / len(total_precision)

        metrics_result = dict()
        metrics_result.update(total_recall)
        metrics_result.update(total_precision)

        log_stats = {split_name: metrics_result}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": metrics_result["total_precision"]}
        return res
