from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
import os
import json
import torch
import numpy as np


@registry.register_task("rag_embedding")
class RAGEmbeddingTask(DeepSpeedBaseTask):

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
        ids, pos_messages, text_types = samples["id"], samples["pos_messages"], samples["text_type"]

        embeddings = model.generate(
            samples,
        )
        for id_, pos, text_type, embedding in zip(ids, pos_messages, text_types, embeddings):
            embedding = embedding.cpu().to(torch.float32).detach().numpy()
            uid = f"{text_type}_{id_}"
            res = {
                "uid": uid,
                "id": id_,
                "text_type": text_type,
            }
            if pos and text_type == "query":
                res["pos"] = pos

            results.append(res)

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
