from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import torch
import torch.nn as nn


@registry.register_task("bengali_asr_infer_task")
class BengaliASRInferTask(BaseTask):
    def __init__(self, evaluate, report_metric=True):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
        )

    @staticmethod
    def calculate_loss(model, logits, attention_mask, labels):
        # retrieve loss input_lengths from attention_mask
        input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=model.config.pad_token_id,
                reduction='none',
                zero_infinity=model.config.ctc_zero_infinity,
            )
        return (loss / target_lengths).detach().cpu().numpy().tolist()

    def valid_step(self, model, samples):
        results = []
        sentences = samples["sentences"]
        ids = samples["ids"]

        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        labels = samples["labels"]
        input_secs = samples["input_secs"]
        model = model.model
        with torch.no_grad():
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            losses = self.calculate_loss(model, logits, attention_mask, labels)

        for loss, sentence, id_, input_sec in zip(losses, sentences, ids, input_secs):
            results.append({
                "loss": loss,
                "sentence": sentence,
                "id": id_,
                "input_sec": input_sec
            })

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics
