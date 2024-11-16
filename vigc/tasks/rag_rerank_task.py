from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
import os
import json
import numpy as np


@registry.register_task("rag_rerank")
class RAGRerankTask(DeepSpeedBaseTask):

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
        ids, queries, passages, prompts, labels = \
            samples["id"], samples["query"], samples["passage"], samples["prompt"], samples["label"]

        scores = model.generate(samples).numpy().tolist()

        for id_, query, passage, prompt, label, score in zip(ids, queries, passages, prompts, labels, scores):
            res = {
                "id": id_,
                "query": query,
                "passage": passage,
                "prompt": prompt,
                "label": label,
                "score": float(score)
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

    @staticmethod
    def _evaluate(
            preds,
            preds_scores,
            labels,
            cutoffs=(1, 10, 25)
    ):
        """
        Evaluate MRR and Recall at cutoffs.
        """
        metrics = {}

        # MRR
        mrrs = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels):
            jump = False
            for i, x in enumerate(pred, 1):
                if x in label:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            mrrs[k] += 1 / i
                    jump = True
                if jump:
                    break
        mrrs /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            mrr = mrrs[i]
            metrics[f"MRR@{cutoff}"] = mrr

        # Recall
        recalls = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels):
            for k, cutoff in enumerate(cutoffs):
                recall = np.intersect1d(label, pred[:cutoff])
                recalls[k] += len(recall) / max(min(cutoff, len(label)), 1)
        recalls /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            recall = recalls[i]
            metrics[f"Recall@{cutoff}"] = recall

        # AUC
        pred_hard_encodings = []
        for pred, label in zip(preds, labels):
            pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
            pred_hard_encodings.append(pred_hard_encoding)

        from sklearn.metrics import roc_auc_score, ndcg_score
        pred_hard_encodings1d = np.asarray(pred_hard_encodings).flatten()
        preds_scores1d = preds_scores.flatten()
        auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)

        metrics['AUC@100'] = auc

        # nDCG
        for k, cutoff in enumerate(cutoffs):
            nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
            metrics[f"nDCG@{cutoff}"] = nDCG

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        with open(eval_result_file) as f:
            results = json.load(f)

        queries = [_['query'] for _ in results]
        passages = [_['passage'] for _ in results]
        labels = [_['label'] for _ in results]
        scores = [_['score'] for _ in results]

        queries_set = list(set(queries))
        pred_info = {_: {"gt": [], "pred": [], "score": []} for _ in queries_set}
        for query, passage, label, score in zip(queries, passages, labels, scores):
            pred_info[query]["pred"].append(passage)
            pred_info[query]["score"].append(score)
            if label:
                pred_info[query]["gt"].append(passage)

        all_preds = [pred_info[_]["pred"] for _ in queries_set]
        all_pred_scores = np.array([pred_info[_]["score"] for _ in queries_set])
        all_gts = [pred_info[_]["gt"] for _ in queries_set]

        cutoffs = [1, 10, 25]
        metrics = self._evaluate(all_preds, all_pred_scores, all_gts, cutoffs=cutoffs)

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in metrics.items()}
        agg_metrics = sum([v for k, v in metrics.items() if k in ("MRR@10",)])
        coco_res["agg_metrics"] = agg_metrics

        return coco_res
