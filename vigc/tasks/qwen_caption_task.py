from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
from vigc.tasks.metrics import Bleu, Rouge
import os
import json


@registry.register_task("qwen_caption")
class QwenCaptionTask(DeepSpeedBaseTask):

    def __init__(self, do_sample, num_beams, max_new_tokens, top_k, top_p, repetition_penalty, temperature, evaluate,
                 report_metric=True):
        super(QwenCaptionTask, self).__init__()
        self.rouge_scorer = Rouge()
        self.bleu_scorer = Bleu()
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        do_sample = generate_cfg.get("do_sample", True)
        num_beams = generate_cfg.get("num_beams", 5)
        max_new_tokens = generate_cfg.get("max_new_tokens", 256)
        top_k = generate_cfg.get("top_k", 20)
        top_p = generate_cfg.get("top_p", 0.8)
        repetition_penalty = generate_cfg.get("repetition_penalty", 1.05)
        temperature = generate_cfg.get("temperature", 0.7)

        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []
        ids, gts, languages = samples["id"], samples["text_output"], samples["language"]

        answers = model.generate(
            samples,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
        )
        for id_, gt, answer, language in zip(ids, gts, answers, languages):
            answer = answer.strip()
            res = {
                "id": id_,
                "gt": gt,
                "pred": answer,
                "language": language,
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

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        metrics = dict()
        with open(eval_result_file) as f:
            results = json.load(f)
        gts = [[_["gt"]] for _ in results]
        preds = [[_["pred"]] for _ in results]

        bleu = self.bleu_scorer.compute_score(gts, preds)
        rouge = self.rouge_scorer.compute_score(gts, preds)

        metrics.update(bleu)
        metrics.update(rouge)

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in metrics.items()}
        agg_metrics = sum([v for k, v in metrics.items() if k in ("Bleu_2", "ROUGE_L")])
        coco_res["agg_metrics"] = agg_metrics

        return coco_res
