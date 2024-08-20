"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
from vigc.common.dist_utils import download_cached_file, main_process
from vigc.common.registry import registry
from vigc.common.utils import is_url
from vigc.runners.runner_iter import RunnerIter


@registry.register_runner("runner_gan_iter")
class RunnerGanIter(RunnerIter):

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is not None:
            return self._optimizer

        gen_num_parameters = 0
        disc_num_parameters = 0
        gen_groups = []
        disc_groups = []
        learning_rate = float(self.config.run_cfg.init_lr)
        weight_decay = float(self.config.run_cfg.weight_decay)
        beta2 = self.config.run_cfg.get("beta2", 0.999)
        beta1 = self.config.run_cfg.get("beta1", 0.9)
        adam_eps = self.config.run_cfg.get("adam_eps", 1e-8)
        for group in self.unwrap_dist_model(self.model).get_generator_parameter_group():
            lr_ratio = group["lr"]
            group_param_nums = group["num_parameters"]
            group_params = group["params"]
            p_wd_gen, p_non_wd_gen = [], []
            for n, p in group_params.items():
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd_gen.append(p)
                else:
                    p_wd_gen.append(p)
            gen_num_parameters += group_param_nums
            this_group_wd = {
                "params": p_wd_gen,
                "lr": lr_ratio * learning_rate,
                "weight_decay": weight_decay
            }
            this_group_non_wd = {
                "params": p_non_wd_gen,
                "lr": lr_ratio * learning_rate,
                "weight_decay": 0
            }
            gen_groups.append(this_group_wd)
            gen_groups.append(this_group_non_wd)
        logging.info("number of generator's trainable parameters: %d" % gen_num_parameters)
        gen_optimizer = torch.optim.AdamW(
            gen_groups,
            lr=float(self.config.run_cfg.init_lr),
            weight_decay=float(self.config.run_cfg.weight_decay),
            betas=(beta1, beta2),
            eps=adam_eps
        )

        for group in self.unwrap_dist_model(self.model).get_discriminator_parameter_group():
            lr_ratio = group["lr"]
            group_param_nums = group["num_parameters"]
            group_params = group["params"]
            p_wd_disc, p_non_wd_disc = [], []
            for n, p in group_params.items():
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd_disc.append(p)
                else:
                    p_wd_disc.append(p)
            disc_num_parameters += group_param_nums
            this_group_wd = {
                "params": p_wd_disc,
                "lr": lr_ratio * learning_rate,
                "weight_decay": weight_decay
            }
            this_group_non_wd = {
                "params": p_non_wd_disc,
                "lr": lr_ratio * learning_rate,
                "weight_decay": 0
            }
            disc_groups.append(this_group_wd)
            disc_groups.append(this_group_non_wd)
        logging.info("number of discriminator's trainable parameters: %d" % disc_num_parameters)
        disc_optimizer = torch.optim.AdamW(
            disc_groups,
            lr=float(self.config.run_cfg.init_lr),
            weight_decay=float(self.config.run_cfg.weight_decay),
            betas=(beta1, beta2),
            eps=adam_eps
        )

        self._optimizer = dict(
            disc=disc_optimizer,
            gen=gen_optimizer,
        )

        return self._optimizer

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is not None:
            return self._lr_sched

        lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

        # max_epoch = self.config.run_cfg.max_epoch
        max_epoch = self.max_epoch
        # min_lr = self.config.run_cfg.min_lr
        min_lr = self.min_lr
        # init_lr = self.config.run_cfg.init_lr
        init_lr = self.init_lr

        # optional parameters
        decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
        warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
        warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
        iters_per_epoch = self.config.run_cfg.get("iters_per_inner_epoch", len(self.train_loader))

        disc_scheduler = lr_sched_cls(
            optimizer=self.optimizer['disc'],
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
            iters_per_epoch=iters_per_epoch,
        )

        gen_scheduler = lr_sched_cls(
            optimizer=self.optimizer['gen'],
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
            iters_per_epoch=iters_per_epoch,
        )

        self._lr_sched = {
            "disc": disc_scheduler,
            "gen": gen_scheduler,
        }
        return self._lr_sched

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

        save_opt_ckpt = self.config.run_cfg.get("save_opt_ckpt", True)
        if save_opt_ckpt:
            save_obj = {
                "model": state_dict,
                "optimizer": {
                    "disc": self.optimizer["disc"].state_dict(),
                    "gen": self.optimizer["gen"].state_dict(),
                },
                "config": self.config.to_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "epoch": cur_iters,
            }
        else:
            save_obj = {
                "model": state_dict,
                "optimizer": None,
                "config": self.config.to_dict(),
                "scaler": None,
                "epoch": cur_iters,
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
        if hasattr(unwrapped_model, "save_checkpoint"):
            save_to = save_to.replace(".pth", "")
            unwrapped_model.save_checkpoint(save_to)
        else:
            torch.save(save_obj, save_to)

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer["disc"].load_state_dict(checkpoint["optimizer"]["disc"])
        self.optimizer["gen"].load_state_dict(checkpoint["optimizer"]["gen"])

        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_iters = checkpoint["iters"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))
