from vigc.common.registry import registry
from vigc.tasks.base_ds_grpo_task import DeepSpeedGRPOBaseTask
from vigc.common.dist_utils import main_process
import os.path as osp
import json
import numpy as np


@registry.register_task("grpo_ds_train")
class GRPODsTask(DeepSpeedGRPOBaseTask):

    def __init__(self, temperature, do_sample, top_p, top_k, repetition_penalty, max_new_tokens, num_beams,
                 evaluate, reward_funcs, num_iterations=1, report_metric=True,
                 agg_metric=""):
        super(GRPODsTask, self).__init__(reward_funcs, num_iterations)
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.evaluate = evaluate
        self.agg_metric = agg_metric
        self.reward_func_dic = {k: registry.get_reward_function(k) for k in reward_funcs}
        assert agg_metric in self.reward_func_dic
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        temperature = generate_cfg.get('temperature', .7)
        do_sample = generate_cfg.get("do_sample", False)
        top_p = generate_cfg.get("top_p", 0.95)
        top_k = generate_cfg.get("top_k", 20)
        repetition_penalty = generate_cfg.get("repetition_penalty", 1.05)
        max_new_tokens = generate_cfg.get("max_new_tokens", 256)
        num_beams = generate_cfg.get("num_beams", 5)

        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        agg_metric = run_cfg.get("agg_metric", "correctness_reward")

        reward_funcs = run_cfg.reward_funcs
        num_iterations = run_cfg.get("num_iterations", 1)

        return cls(
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            evaluate=evaluate,
            report_metric=report_metric,
            agg_metric=agg_metric,
            reward_funcs=reward_funcs,
            num_iterations=num_iterations,
        )

    def valid_step(self, model, samples):
        results = []
        text_inputs, text_outputs, ids = samples["text_input"], samples["text_output"], samples["id"]
        preds = model.generate(
            samples,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
        )

        for text_input, text_output, pred, id_ in zip(text_inputs, text_outputs, preds, ids):
            rewards = {k: v(text_input, text_output, pred) for k, v in self.reward_func_dic.items()}

            this_item = {
                "input_text": text_input,
                "gt": text_output,
                "pred": pred,
                "rewards": rewards,
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

        reward_lst = [_["rewards"] for _ in results]
        reward_keys = reward_lst[0].keys()
        reward_dic = {k: [_[k] for _ in reward_lst] for k in reward_keys}
        eval_ret = {k: np.mean(v) for k, v in reward_dic.items()}
        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                osp.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in eval_ret.items()}
        coco_res["agg_metrics"] = eval_ret[self.agg_metric]

        return coco_res
