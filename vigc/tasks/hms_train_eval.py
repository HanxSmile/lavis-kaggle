from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np
import pandas as pd
from .kaggle_kl_div import score


@registry.register_task("hms_train_eval")
class HMSClassifyTrainEvalTask(BaseTask):

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
        probs, label, eeg_id = response["result"], response["label"], response["eeg_id"]
        for eeg_id_, prob_, label_ in zip(eeg_id, probs, label):
            this_res = {
                "eeg_id": str(eeg_id_),
                "prob": prob_.tolist(),
                "label": label_.tolist()
            }
            results.append(this_res)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="eeg_id",
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

        pred_buffer = list()
        label_buffer = list()

        for result in results:
            pred = result["prob"]
            label = result["label"]
            pred_buffer.append(pred)
            label_buffer.append(label)

        all_oof = np.array(pred_buffer)
        all_true = np.array(label_buffer)
        oof = pd.DataFrame(all_oof.copy())
        oof['id'] = np.arange(len(oof))

        true = pd.DataFrame(all_true.copy())
        true['id'] = np.arange(len(true))
        cv = float(score(solution=true, submission=oof, row_id_column_name='id'))
        log_stats = {split_name: {"kl-div": cv}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": 10 - cv}
        return res
