from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.gan_base_task import GanBaseTask
import os
import os.path as osp
import scipy
import jiwer
import json
import numpy as np


@registry.register_task("tts_task")
class TTSTask(GanBaseTask):

    def __init__(self, evaluate, report_metric=True, metric_key=None):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.epoch = 0
        self.metric_key = metric_key

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        evaluate = run_cfg.evaluate
        metric_key = run_cfg.get("metric_key", None)

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            metric_key=metric_key,
        )

    def valid_step(self, model, samples):
        results = []
        texts = samples["texts"]
        ids = samples["ids"]

        preds = model.generate(
            samples,
            return_loss=True
        )
        save_root = registry.get_path("result_dir")
        for text, pred, id_ in zip(texts, preds, ids):
            save_dir = osp.join(save_root, f"audios-epoch-{self.epoch}")
            if not osp.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            audio = pred["audio"]
            sampling_rate = pred["sampling_rate"]
            save_path = osp.join(save_dir, f"{id_}.wav")
            scipy.io.wavfile.write(save_path, rate=sampling_rate, data=audio)
            res = {
                "text": text,
                "id": id_,
                "audio": save_path,
            }
            if "pred" in pred:
                res["pred"] = pred["pred"]
                res["wer"] = jiwer.wer(res["text"], res["pred"]) * 100
                res["cer"] = jiwer.cer(res["text"], res["pred"]) * 100
            results.append(res)
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
        self.epoch += 1
        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        with open(eval_result_file) as f:
            results = json.load(f)
        if "cer" not in results[0]:
            return {"agg_metrics": 0.0}

        wers = [_["wer"] for _ in results]
        cers = [_["cer"] for _ in results]

        wer = np.mean(wers)
        cer = np.mean(cers)

        res_info = dict(cer=cer, wer=wer)

        log_stats = {split_name: res_info}
        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")
        if self.metric_key == "wer":
            res = {"agg_metrics": 100 - wer, "wer": wer}
        elif self.metric_key == "cer":
            res = {"agg_metrics": 100 - cer, "cer": cer}
        else:
            res = {"agg_metrics": 0.0}
        return res
