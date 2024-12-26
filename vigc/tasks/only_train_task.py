from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask


@registry.register_task("only_train_task")
class OnlyTrainTask(BaseTask):

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
            report_metric=report_metric
        )

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
