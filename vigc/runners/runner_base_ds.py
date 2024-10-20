import logging
import os
from pathlib import Path
from omegaconf import OmegaConf

import torch
from vigc.common.registry import registry
from vigc.runners.runner_base import RunnerBase

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
    def use_distributed(self):
        return True

    def _init_ds_module(self):
        self._wrapped_model, self._wrapped_optimizer, _, _ = deepspeed.initialize(
            model=self._model,
            optimizer=self._optimizer,
            lr_scheduler=None,
            config=OmegaConf.to_container(self.deepspeed_config),
        )
        self._wrapped_model.save_fp16_model()

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
        self.model.save_checkpoint(self.deepspeed_ckpt_dir, tag=save_tag, client_state=client_sd)

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
