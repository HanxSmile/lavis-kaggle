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
        langs = samples.get("languages", [None] * len(ids))
        preds, losses = model.generate(
            samples,
            return_loss=True
        )
        for gt, pred, loss, id_, lang_ in zip(gts, preds, losses, ids, langs):
            results.append({
                "loss": float(loss),
                "wer": float(100 * jiwer.wer(gt, pred)),
                "cer": float(100 * jiwer.cer(gt, pred)),
                "gt": gt,
                "pred": pred,
                "id": id_,
                "language": lang_
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
        langs = [_["language"] for _ in results]

        res_info = dict()
        if langs[0] is None:
            loss = np.mean(losses)
            wer = 100 * jiwer.wer(gts, preds)
            cer = 100 * jiwer.cer(gts, preds)
            res_info["wer"] = wer
            res_info["cer"] = cer
            res_info["loss"] = loss
        else:
            stats_info = dict()
            for lang_, gt_, pred_, loss_ in zip(langs, gts, preds, losses):
                lang_info = stats_info.setdefault(lang_, dict())
                gt_info = lang_info.setdefault("gt", list())
                pred_info = lang_info.setdefault("pred", list())
                loss_info = lang_info.setdefault("loss", list())
                gt_info.append(gt_)
                pred_info.append(pred_)
                loss_info.append(loss_)
            for lang_, lang_info in stats_info.items():
                gts_, preds_, losses_ = lang_info["gt"], lang_info["pred"], lang_info["loss"]
                wer = 100 * jiwer.wer(gts_, preds_)
                cer = 100 * jiwer.cer(gts_, preds_)
                loss_ = np.mean(losses_)
                res_info[f"{lang_}_wer"] = wer
                res_info[f"{lang_}_cer"] = cer
                res_info[f"{lang_}_loss"] = loss_
        log_stats = {split_name: res_info}
        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")
        if self.metric_key == "wer":
            wers = [v for k, v in res_info.items() if "wer" in k]
            res = {"agg_metrics": 100 - np.mean(wers), "wer": np.mean(wers)}
        elif self.metric_key == "cer":
            cers = [v for k, v in res_info.items() if "cer" in k]
            res = {"agg_metrics": 100 - np.mean(cers), "cer": np.mean(cers)}
        else:
            losses = [v for k, v in res_info.items() if "loss" in k]
            res = {"agg_metrics": 10 - np.mean(losses), "loss": np.mean(losses)}
        return res
