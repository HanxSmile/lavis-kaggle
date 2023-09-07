from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import os
import torch.distributed as dist
import logging


@registry.register_task("indic_corp_infer_task")
class BengaliIndicCorpInferTask(BaseTask):
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

    def valid_step(self, model, samples):
        return samples

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )

        metrics = {"agg_metrics": 0.0}

        return metrics

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):

        result_file = os.path.join(
            result_dir, "%s_rank%d.txt" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.txt" % filename)
        with open(result_file, "w") as f:
            for sentence in result:
                f.write(sentence)
                f.write(" ")

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes

            with open(final_result_file, "w") as wf:
                for rank in range(get_world_size()):
                    result_file = os.path.join(
                        result_dir, "%s_rank%d.txt" % (filename, rank)
                    )
                    with open(result_file) as rf:
                        sentence = rf.readline().strip()
                    if sentence:
                        wf.write(sentence)
                        wf.write(" ")

            print("result file saved to %s" % final_result_file)

        return final_result_file
