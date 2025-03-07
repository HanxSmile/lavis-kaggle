from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np


@registry.register_task("image_ctr_task")
class ImageCTRTask(BaseTask):

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

        preds = model.generate(samples).cpu().numpy().tolist()
        labels = samples["label"].cpu().numpy().tolist()
        ids = samples["id"]

        for pred_, label_, id_ in zip(preds, labels, ids):
            this_res = {
                "pred": float(pred_),
                "label": int(label_),
                "id": id_,
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

    @staticmethod
    def inversion_pair_proportion(y_true, y_pred):
        n = len(y_true)
        inversion_count = 0
        pair_count = 0

        # 遍历所有的样本对 (i, j)，其中 i < j
        for i in range(n):
            for j in range(i + 1, n):
                # 如果真实值相同，则跳过此对
                if y_true[i] == y_true[j]:
                    continue

                pair_count += 1
                # 如果预测值和真实值的顺序相反，则计为逆序
                if (y_pred[i] > y_pred[j] and y_true[i] < y_true[j]) or (
                        y_pred[i] < y_pred[j] and y_true[i] > y_true[j]):
                    inversion_count += 1

        # 计算逆序对比例
        return inversion_count / pair_count if pair_count != 0 else 0

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2))

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        with open(eval_result_file) as f:
            results = json.load(f)

        preds = [_["pred"] for _ in results]
        labels = [_["label"] for _ in results]
        rmse = self.rmse(labels, preds)
        inv_prop = self.inversion_pair_proportion(labels, preds)
        log_stats = {split_name: {"rmse": rmse, "inv_prop": inv_prop}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": 1 - inv_prop}
        return res
