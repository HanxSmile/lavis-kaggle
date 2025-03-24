from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
from vigc.tasks.metrics import Bleu, Rouge
from sacrebleu import corpus_bleu, sentence_bleu
import os
import re
import json
import jieba


def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text


def metric_post_process(text):
    text = pre_clean(text)
    text = text.replace('#', ' ')
    return text


def get_zh_text(text):
    split_lines = []
    lines = text.split('\n')
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)


@registry.register_task("qwen_translation")
class QwenTranslationTask(DeepSpeedBaseTask):
    """
    Chinese translation task of Qwen.
    """

    def __init__(self, do_sample, num_beams, max_new_tokens, top_k, top_p, repetition_penalty, temperature, evaluate,
                 report_metric=True):
        super(QwenTranslationTask, self).__init__()
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
        ids, gts, questions, system_inputs = samples["id"], samples["text_output"], \
            samples["text_input"], samples["system_input"]

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
        for id_, gt, answer, question, system_input in zip(
                ids, gts, answers, questions, system_inputs):
            answer = answer.strip()
            res = {
                "id": id_,
                "gt": gt,
                "pred": answer,
                "question": question,
                "system_input": system_input,
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

        with open(eval_result_file) as f:
            results = json.load(f)
        gts = [metric_post_process(get_zh_text(_['gt'])) for _ in results]
        preds = [metric_post_process(get_zh_text(_['pred'])) for _ in results]
        bleu_score = corpus_bleu(preds, [gts]).score
        metrics = dict(bleu=bleu_score)

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in metrics.items()}
        coco_res["agg_metrics"] = bleu_score

        return coco_res
