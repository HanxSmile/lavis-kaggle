"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import random
import logging
import torch
from vigc.common.registry import registry
from vigc.common.logger import MetricLogger, SmoothedValue
from vigc.datasets.data_utils import prepare_sample
from vigc.tasks.utils.deepspeed_utils import unwrap_model_for_generation
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import is_dist_avail_and_initialized
import torch.distributed as dist


class DeepSpeedGRPOBaseTask(BaseTask):
    def __init__(self, reward_funcs, num_iterations=1, ds3_gather_for_generation=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        self.num_iterations = num_iterations
        self.reward_funcs = {k: registry.get_reward_function(k) for k in reward_funcs}
        self.reward_func_keys = sorted(self.reward_funcs.keys())
        self.ds3_gather_for_generation = ds3_gather_for_generation or False

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        with unwrap_model_for_generation(model, self.ds3_gather_for_generation) as unwrapped_model:

            for samples in metric_logger.log_every(data_loader, print_freq, header):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=unwrapped_model, samples=samples)
                results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def _train_inner_loop(
            self,
            epoch,
            iters_per_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=True,
            start_iters=None,
            log_freq=50,
            accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        grpo_inputs = None
        num_generations = 0

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            if grpo_inputs is None:
                samples = next(data_loader)
                samples = prepare_sample(samples)
                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            model_dtype = next(model.parameters()).dtype

            # prepare samples
            with (torch.cuda.amp.autocast(dtype=model_dtype,
                                          cache_enabled=False) if model_dtype != torch.float32 else contextlib.nullcontext()):
                if grpo_inputs is None:
                    # TODO: consider ZERO-3
                    with unwrap_model_for_generation(model, self.ds3_gather_for_generation) as unwrapped_model:
                        this_grpo_inputs = unwrapped_model(samples=samples, prepare_inputs_flag=True,
                                                           reward_funcs=self.reward_funcs)
                    grpo_inputs = {}
                    num_generations = len(this_grpo_inputs["llm_inputs"])
                    group_index = list(range(num_generations))
                    for _ in range(self.num_iterations):
                        random.shuffle(group_index)
                        for k, v in this_grpo_inputs.items():
                            v = [v[_] for _ in group_index]
                            grpo_inputs[k] = grpo_inputs.get(k, []) + v
                    dist.barrier()

                input_idx = 0
                model_input = {k: v[input_idx] for k, v in grpo_inputs.items()}
                for k, v in grpo_inputs.items():
                    v.pop(input_idx)
                if len(grpo_inputs["llm_inputs"]) == 0:
                    grpo_inputs = None
                output = model(**model_input, prepare_inputs_flag=False)
                rewards_info = output.pop("rewards_info")
                rewards_info = {k: rewards_info[i] for i, k in enumerate(self.reward_func_keys)}
                output.update(rewards_info)

                loss_dict = {}
                for k, v in output.items():
                    if "loss" in k or "reward" in k:
                        loss_dict[k] = v.detach().clone()  # not affect loss_dict values for logging
                loss = output["loss"]

            # after_train_step()
            model.backward(loss)
            model.step()

            # update gradients every accum_grad_iters iterations
            # now don't need
            if (i + 1) % accum_grad_iters == 0:
                pass

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
