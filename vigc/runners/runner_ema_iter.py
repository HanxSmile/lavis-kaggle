"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from ema_pytorch import EMA
from vigc.runners.runner_iter import RunnerIter


@registry.register_runner("runner_ema_iter")
class RunnerEmaIter(RunnerIter):
    """
    Run training based on the number of iterations. This is common when
    the training dataset size is large. Underhood logic is similar to
    epoch-based training by considering every #iters_per_inner_epoch as an
    inner epoch.

    In iter-based runner, after every #iters_per_inner_epoch steps, we

        1) do a validation epoch;
        2) schedule the learning rate;
        3) save the checkpoint.

    We refer every #iters_per_inner_epoch steps as an inner epoch.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self._ema_model = EMA(
            model,
            ema_model=task.build_model(cfg),
            beta=cfg.run_cfg.get("ema_decay", 0.995),
            update_every=cfg.run_cfg.get("ema_update_every", 10)).to(self.device)

    def train_iters(self, epoch, start_iters):
        # train by iterations
        self.model.train()

        return self.task.train_iters(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_inner_epoch=self.iters_per_inner_epoch,
            ema=self._ema_model,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self._ema_model.ema_model
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    @main_process
    def _save_checkpoint(self, cur_iters, is_best=False, latest=False):
        # only save the params requires gradient
        assert not (is_best and latest), "You can't set 'is_best' and 'latest' the same time."
        unwrapped_model = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in unwrapped_model.named_parameters()
        }

        state_dict = unwrapped_model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]

        save_obj = {
            "model": self._ema_model.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "iters": cur_iters,
        }
        if is_best:
            save_to = os.path.join(
                self.output_dir,
                "checkpoint_{}.pth".format("best"),
            )
        elif latest:
            save_to = os.path.join(
                self.output_dir,
                "checkpoint_{}.pth".format("latest"),
            )
        else:
            save_to = os.path.join(
                self.output_dir,
                "checkpoint_{}.pth".format(cur_iters),
            )
        logging.info("Saving checkpoint at iters {} to {}.".format(cur_iters, save_to))
        torch.save(save_obj, save_to)
