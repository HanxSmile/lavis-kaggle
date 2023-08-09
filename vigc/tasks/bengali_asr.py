from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import json
from bnunicodenormalizer import Normalizer
import jiwer

bnorm = Normalizer()


def normalize(sen):
    _words = [bnorm(word)['normalized'] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def dari(sentence):
    try:
        if sentence[-1] != "।":
            sentence += "।"
    except:
        print(sentence)
    return sentence


@registry.register_task("whisper_bengali_asr_task")
class WhisperBengaliASRTask(BaseTask):

    def __init__(self, evaluate, report_metric=True, post_process_flag=True):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.post_process_flag = post_process_flag

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        post_process_flag = cfg.model_cfg.get("post_process_flag", True)
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            post_process_flag=post_process_flag
        )

    def valid_step(self, model, samples):
        results = []
        gts = samples["sentences"]
        ids = samples["ids"]
        if self.post_process_flag:
            gts = [dari(normalize(_)) for _ in gts]
        preds = model.generate(
            samples
        )
        for gt, pred, id_ in zip(gts, preds, ids):
            results.append({
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
        wer = 100 * jiwer.wer(gts, preds)
        log_stats = {split_name: {"wer": wer}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": 100 - wer, "wer": wer}
        return res
