import re
from vigc.common.registry import registry
from vigc.tasks.base_ds_task import DeepSpeedBaseTask
from vigc.common.dist_utils import main_process
import os.path as osp
import json
import jieba
from sacrebleu import corpus_bleu, sentence_bleu
from zss import simple_distance, Node


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


def clean_table_math(text):
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', text)
    text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', text)
    text = re.sub(r'\\\(.*?\\\)', ' ', text)
    text = re.sub(r'\\\[.*?\\\]', ' ', text)
    return text


def get_zh_text(text):
    split_lines = []
    lines = text.split('\n')
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)


def get_tree(text):
    tree = (Node('ROOT').addkid(Node('TITLE')))
    mmd_lines = text.split('\n')
    lines = []
    for line in mmd_lines:
        line = pre_clean(line)
        if line.strip() != '':
            lines.append(line.strip())
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree


def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0

    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1 - total_distance / num_of_nodes


@registry.register_task("ds_zh_img2mkdown_train")
class DeepSpeedZhImg2MarkdownTask(DeepSpeedBaseTask):

    def __init__(self, temperature, do_sample, top_p, top_k, num_beams, max_new_tokens, repetition_penalty, evaluate,
                 report_metric=True, agg_metric="bleu"):
        super(DeepSpeedZhImg2MarkdownTask, self).__init__()
        self.temperature = temperature
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p

        self.evaluate = evaluate
        self.agg_metric = agg_metric

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        temperature = generate_cfg.get('temperature', .2)
        do_sample = generate_cfg.get("do_sample", False)
        top_p = generate_cfg.get("top_p", 0.95)
        top_k = generate_cfg.get("top_k", 20)
        num_beams = generate_cfg.get("num_beams", 5)
        max_new_tokens = generate_cfg.get("max_new_tokens", 1500)
        repetition_penalty = generate_cfg.get("repetition_penalty", 1.05)

        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        agg_metric = run_cfg.get("agg_metric", "bleu")

        return cls(
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            evaluate=evaluate,
            report_metric=report_metric,
            agg_metric=agg_metric,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

    def valid_step(self, model, samples):
        results = []
        text, ids = samples["text_output"], samples["id"]
        pred_strs = model.generate(
            samples,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
        )
        truth_strs = text

        for pred_str, truth_str, id_ in zip(pred_strs, truth_strs, ids):
            pred_raw_str = pred_str
            pred_clean_str = metric_post_process(get_zh_text(pred_str))
            pred_clean_table_math_str = clean_table_math(pred_clean_str)

            truth_raw_str = truth_str
            truth_clean_str = metric_post_process(get_zh_text(truth_str))
            truth_clean_table_math_str = clean_table_math(truth_clean_str)

            this_item = {
                "pred_raw_str": pred_raw_str,
                "pred_clean_str": pred_clean_str,
                "pred_clean_table_math_str": pred_clean_table_math_str,
                "truth_raw_str": truth_raw_str,
                "truth_clean_str": truth_clean_str,
                "truth_clean_table_math_str": truth_clean_table_math_str,
                "id": id_
            }
            results.append(this_item)
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

        pred_raw_str_lst = [_["pred_raw_str"] for _ in results]
        pred_tree_lst = [get_tree(_) for _ in pred_raw_str_lst]
        pred_clean_str_lst = [_["pred_clean_str"] for _ in results]
        pred_clean_table_math_str_lst = [_["pred_clean_table_math_str"] for _ in results]

        truth_raw_str_lst = [_["truth_raw_str"] for _ in results]
        truth_tree_lst = [get_tree(_) for _ in truth_raw_str_lst]
        truth_clean_str_lst = [_["truth_clean_str"] for _ in results]
        truth_clean_table_math_str_lst = [_["truth_clean_table_math_str"] for _ in results]

        steds_lst = [STEDS(pred_tree, ref_tree) for pred_tree, ref_tree in zip(pred_tree_lst, truth_tree_lst)]

        bleu_score = corpus_bleu(pred_clean_str_lst, [truth_clean_str_lst]).score * 100
        bleu_pt_score = corpus_bleu(pred_clean_table_math_str_lst, [truth_clean_table_math_str_lst]).score * 100
        steds_score = sum(steds_lst) / len(steds_lst) * 100

        eval_ret = {"bleu": bleu_score, "bleu_pt": bleu_pt_score, "steds": steds_score}

        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                osp.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in eval_ret.items()}

        if "bleu_pt" in self.agg_metric.lower():
            agg_metrics = bleu_pt_score
        elif "bleu" in self.agg_metric.lower():
            agg_metrics = bleu_score
        elif "sted" in self.agg_metric.lower():
            agg_metrics = steds_score
        else:
            raise ValueError(f"Invalid metrics: '{self.agg_metric}'")

        coco_res["agg_metrics"] = agg_metrics

        return coco_res
