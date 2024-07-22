from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
import jiwer
import numpy as np


@registry.register_task("whisper_asr_task")
class WhisperASRTask(BaseTask):

    def __init__(self, evaluate, report_metric=True, metric_key="loss"):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.metric_key = metric_key

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        metric_key = run_cfg.get("metric_key", "loss")
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            metric_key=metric_key,
        )

    def valid_step(self, model, samples):
        results = []
        gts = samples["sentences"]
        ids = samples["ids"]
        preds, losses = model.generate(
            samples,
            return_loss=True
        )
        for gt, pred, loss, id_ in zip(gts, preds, losses, ids):
            results.append({
                "loss": float(loss),
                "wer": float(100 * jiwer.wer(gt, pred)),
                "gt": gt,
                "pred": pred,
                "id": id_
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
        losses = [_["loss"] for _ in results]
        loss = np.mean(losses)
        wer = 100 * jiwer.wer(gts, preds)
        log_stats = {split_name: {"wer": wer, "loss": loss}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")
        if self.metric_key == "wer":
            res = {"agg_metrics": 100 - wer, "wer": wer}
        else:
            res = {"agg_metrics": 10 - loss, "loss": loss}
        return res
