import logging
import os
import time
import datetime
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from vigc.common.registry import registry
from vigc.runners.runner_base import RunnerBase
from vigc.common.dist_utils import (
    get_world_size,
    is_main_process,
)
import deepspeed


@registry.register_runner("runner_base_ds")
class DeepSpeedRunner(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
        self._wrapped_optimizer = None
        self._optimizer = self._init_optimizer()
        self.zero_stage = self.deepspeed_config.zero_optimization.stage
        # assert self.zero_stage !=3, 'Not support zero3'

    @property
    def deepspeed_config(self):
        return self.config.run_cfg.deepspeed_config

    @property
    def only_save_latest(self):
        only_save_latest = self.config.run_cfg.get("only_save_latest", False)
        only_save_best = self.config.run_cfg.get("only_save_best", False)
        assert only_save_latest + only_save_best <= 1
        return only_save_latest

    @property
    def only_save_best(self):
        only_save_latest = self.config.run_cfg.get("only_save_latest", False)
        only_save_best = self.config.run_cfg.get("only_save_best", False)
        assert only_save_latest + only_save_best <= 1
        return only_save_best

    @property
    def use_distributed(self):
        return True

    def _init_ds_module(self):
        self._wrapped_model, self._wrapped_optimizer, _, _ = deepspeed.initialize(
            model=self._model,
            optimizer=self._optimizer,
            lr_scheduler=None,
            config=OmegaConf.to_container(self.deepspeed_config),
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
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(self.model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )


    @property
    def model(self):
        """
        DeepSpeed wrapped.
        """
        if self._wrapped_model is None:
            self._init_ds_module()
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._wrapped_optimizer is None:
            self._init_ds_module()
        return self._wrapped_optimizer

    def _init_optimizer(self):
        if self._optimizer is not None:
            return self._optimizer

        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        logging.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.config.run_cfg.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = self.config.run_cfg.get("beta2", 0.999)
        self._optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.config.run_cfg.init_lr),
            weight_decay=float(self.config.run_cfg.weight_decay),
            betas=(0.9, beta2),
        )
        return self._optimizer

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    best_flag = False
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                    "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "eval":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                best_flag = True
                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
                    if best_flag:
                        best_flag_tensor = torch.ones(1).to(self.device)
                    else:
                        best_flag_tensor = torch.zeros(1).to(self.device)

                    flag_tensors = torch.zeros(get_world_size()).to(self.device)
                    dist.all_gather_into_tensor(flag_tensors, best_flag_tensor)
                    if torch.sum(flag_tensors) > 0 and (not self.only_save_latest):
                        self._save_checkpoint(cur_epoch, is_best=True)

            if self.evaluate_only:
                break
            if self.milestone and cur_epoch + 1 in self.milestone and (not self.only_save_latest) and (
                    not self.only_save_best):
                self._save_checkpoint(cur_epoch)
            if not self.only_save_best:
                self._save_checkpoint(cur_epoch, latest=True)
            dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"
        deepspeed_ckpt_dir = output_dir / "deepspeed_ckpt"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))
        registry.register_path("deepspeed_ckpt_dir", str(deepspeed_ckpt_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir
        self.deepspeed_ckpt_dir = deepspeed_ckpt_dir

    # deepspeed save_checkpoint cannot work only on rank0 otherwise hanging the other processes
    # @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False, latest=False):
        """
        Save the checkpoint at the current epoch.
        """
        assert not (is_best and latest), "You can't set 'is_best' and 'latest' the same time."
        # assert self.zero_stage != 3, 'not support zero3'
        client_sd = {
            "config": self.config.to_dict(),
            "epoch": cur_epoch,
        }
        if is_best:
            save_tag = "checkpoint_{}".format("best")
        elif latest:
            save_tag = "checkpoint_{}".format("latest")
        else:
            save_tag = "checkpoint_{}".format(cur_epoch)
        logging.info(
            "Saving checkpoint at epoch {} to {}.".format(cur_epoch, os.path.join(self.deepspeed_ckpt_dir, save_tag)))
        self.model.save_checkpoint(self.deepspeed_ckpt_dir, tag=save_tag, client_state=client_sd,
                                   exclude_frozen_parameters=True)

    # 暂时没经过严格测试，应该会有问题
    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        print("该函数暂时没经过严格测试，应该会有问题")
        checkpoint_path = os.path.join(self.deepspeed_ckpt_dir, "checkpoint_best")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'), map_location="cpu")
        try:
            model.load_state_dict(checkpoint["module"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["module"], strict=False)
        return model

    def _load_checkpoint(self, load_dir, tag=None):
        """
        Resume from a checkpoint.
        """
        if tag is None:
            load_dir, tag = load_dir.split(":")[-2:]
        if os.path.isdir(load_dir):
            _, client_sd = self.model.load_checkpoint(load_dir=load_dir, tag=tag)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        self.start_epoch = client_sd["epoch"] + 1
        logging.info("Resume checkpoint from {}/{}".format(load_dir, tag))
