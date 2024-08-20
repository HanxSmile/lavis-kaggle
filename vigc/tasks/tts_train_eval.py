from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.gan_base_task import GanBaseTask
import os
import os.path as osp
import scipy


@registry.register_task("tts_task")
class TTSTask(GanBaseTask):

    def __init__(self, evaluate, report_metric=True):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.epoch = 0

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
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
        return {"agg_metrics": 0.0}
