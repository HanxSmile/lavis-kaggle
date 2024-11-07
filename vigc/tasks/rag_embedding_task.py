from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import random


@registry.register_task("rag_embedding")
class RAGEmbeddingTask(DeepSpeedBaseTask):

    def __init__(self, evaluate, report_metric=True, range_for_sampling=(10, 210), negative_number=15):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.range_for_sampling = range_for_sampling
        self.negative_number = negative_number

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.get("generate_cfg")
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)

        range_for_sampling = generate_cfg.range_for_sampling
        range_for_sampling = [int(_) for _ in range_for_sampling.split("-")]
        assert len(range_for_sampling) == 2

        negative_number = int(generate_cfg.negative_number)

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
            range_for_sampling=range_for_sampling,
            negative_number=negative_number,
        )

    def valid_step(self, model, samples):
        results = []
        ids, pos_messages, text_types, texts = samples["id"], samples["pos_messages"], samples["text_type"], samples[
            "original_text"]

        embeddings = model.generate(
            samples,
        )
        save_root = registry.get_path("result_dir")
        passage_root = os.path.join(save_root, "passages")
        query_root = os.path.join(save_root, "queries")
        if not os.path.isdir(passage_root):
            os.makedirs(passage_root, exist_ok=True)
        if not os.path.isdir(query_root):
            os.makedirs(query_root, exist_ok=True)
        for id_, pos, text_type, embedding, text in zip(ids, pos_messages, text_types, embeddings, texts):
            embedding = embedding.cpu().to(torch.float32).detach().numpy()
            if text_type == "query":
                embedding_path = os.path.join(query_root, f"{id_}.npy")
            else:
                embedding_path = os.path.join(passage_root, f"{id_}.npy")
            np.save(embedding_path, embedding[None])
            uid = f"{text_type}_{id_}"
            res = {
                "uid": uid,
                "id": id_,
                "text": text,
                "text_type": text_type,
                "embedding": embedding_path
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

    @staticmethod
    def _evaluate(
            preds,
            preds_scores,
            labels,
            cutoffs=[1, 10, 100]
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

        passages = [_ for _ in results if _["text_type"] == "passage"]
        queries = [_ for _ in results if _["text_type"] == "query"]
        passage_dic = {
            _["id"]: _ for _ in passages
        }
        passage_ids = [_["id"] for _ in passages]
        passage_embeddings = [torch.from_numpy(np.load(_["embedding"])) for _ in passages]
        passage_embeddings = torch.cat(passage_embeddings, dim=0)

        all_preds = []
        all_pred_scores = []
        all_gts = []

        results_for_train = []
        for query_info in queries:
            query_embedding = torch.from_numpy(np.load(query_info["embedding"]))
            score = F.cosine_similarity(query_embedding, passage_embeddings)
            all_sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()
            sort_index = all_sort_index[:100]
            preds = [passage_ids[_] for _ in sort_index]
            scores = np.array([score[_].item() for _ in sort_index])[None]
            gts = [str(_) for _ in query_info["pos"]]
            all_preds.append(preds)
            all_pred_scores.append(scores)
            all_gts.append(gts)

            index_for_hn_mine = all_sort_index[self.range_for_sampling[0]: self.range_for_sampling[1]]
            index_for_hn_mine = [passage_ids[_] for _ in index_for_hn_mine if passage_ids[_] not in gts]

            if len(index_for_hn_mine) > self.negative_number:
                index_for_hn_mine = random.sample(index_for_hn_mine, self.negative_number)

            this_item = {
                "query": query_info["text"],
                "pos": [passage_dic[_]['text'] for _ in gts],
                "neg": [passage_dic[_]['text'] for _ in index_for_hn_mine],
            }
            results_for_train.append(this_item)

        cutoffs = [1, 10, 100]
        metrics = self._evaluate(all_preds, np.concatenate(all_pred_scores, axis=0), all_gts, cutoffs=cutoffs)

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        with open(
                os.path.join(registry.get_path("output_dir"), "hn_min_result.json"), "w"
        ) as f:
            json.dump(results_for_train, f)

        coco_res = {k: v for k, v in metrics.items()}
        agg_metrics = sum([v for k, v in metrics.items() if k in ("MRR@10",)])
        coco_res["agg_metrics"] = agg_metrics

        return coco_res
