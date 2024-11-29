from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
import os
import json
import sacrebleu


@registry.register_task("translation_ds_task")
class TranslationDeepSpeedTask(DeepSpeedBaseTask):

    def __init__(self, evaluate, report_metric=True, num_beams=5, return_loss=False):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.num_beams = num_beams
        self.return_loss = return_loss

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        generate_cfg = run_cfg.get("generate_cfg")

        num_beams = generate_cfg.get("num_beams", 5)
        return_loss = generate_cfg.get("return_loss", False)

        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            num_beams=num_beams,
            return_loss=return_loss
        )

    def valid_step(self, model, samples):
        results = []
        gts = samples["output"]
        inputs = samples["input"]
        ids = samples["ids"]
        preds, loss_lst = model.generate(
            samples,
            num_beams=self.num_beams,
            return_loss=self.return_loss
        )
        for gt, pred, id_, loss_, input_ in zip(gts, preds, ids, loss_lst, inputs):
            score = sacrebleu.corpus_bleu([pred], [gt]).score
            results.append({
                "input": input_,
                "gt": gt,
                "pred": pred,
                "id": id_,
                "loss": loss_,
                "bleu": score
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
        gts = [[_["gt"] for _ in results]]
        preds = [_["pred"] for _ in results]
        bleu = sacrebleu.corpus_bleu(preds, gts).score
        bleu = float(bleu)
        log_stats = {split_name: {"bleu": bleu}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": bleu, "bleu": bleu}
        return res
