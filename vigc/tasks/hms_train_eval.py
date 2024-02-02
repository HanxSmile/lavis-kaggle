from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import numpy as np
import torch.nn as nn
import torch


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
        eeg_id, spec_id, uid = response["eeg_id"], response["spec_id"], response["uid"]
        probs, label = response["probs"], response["label"]
        for eeg_id_, spec_id_, uid_, prob_, label_ in zip(eeg_id, spec_id, uid, probs, label):
            this_res = {
                "eeg_id": str(eeg_id_),
                "spec_id": str(spec_id_),
                "uid": str(uid_),
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
            remove_duplicate="uid",
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

        prob_buffer = dict()
        label_buffer = dict()
        count_buffer = dict()
        for result in results:
            eeg_id = result["eeg_id"]
            prob = np.array(result["prob"])
            label = np.array(result["label"])

            prob_buffer[eeg_id] = prob + prob_buffer.get(eeg_id, 0)
            label_buffer[eeg_id] = label + label_buffer.get(eeg_id, 0)
            count_buffer[eeg_id] = 1 + count_buffer.get(eeg_id, 0)

        for eeg_id in prob_buffer:
            prob_buffer[eeg_id] = torch.from_numpy(prob_buffer[eeg_id] / count_buffer[eeg_id])
            label_buffer[eeg_id] = torch.from_numpy(label_buffer[eeg_id] / count_buffer[eeg_id])

        all_probs = []
        all_labels = []
        for eeg_id in prob_buffer:
            all_probs.append(prob_buffer[eeg_id])
            all_labels.append(label_buffer[eeg_id])

        all_probs = torch.log(torch.stack(all_probs, dim=0))
        all_labels = torch.stack(all_labels, dim=0)

        loss = nn.KLDivLoss(reduction="batchmean")(all_probs, all_labels)

        log_stats = {split_name: {"loss": loss}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": 10 - loss}
        return res
