from vigc.tasks.base_task import BaseTask
import torch
import logging
from vigc.common.logger import MetricLogger, SmoothedValue
from vigc.datasets.data_utils import prepare_sample


class GanBaseTask(BaseTask):
    def _train_inner_loop(
            self,
            epoch,
            iters_per_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            start_iters=None,
            log_freq=50,
            cuda_enabled=False,
            accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        grad_norm = 1.0  # TODO: read from config
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        # metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("gen_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("disc_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

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

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler['disc'].step(cur_epoch=inner_epoch, cur_step=i)
            lr_scheduler['gen'].step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                common_outputs = model("common", samples)
                # -----------------------
                #  Train Discriminator
                # -----------------------
                disc_loss = model("discriminator", *common_outputs)["loss"]
                disc_loss /= accum_grad_iters
            if use_amp:
                scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                disc_params = [p for n, p in model.named_parameters() if ".discriminator." in n]
                if use_amp:
                    scaler.unscale_(optimizer['disc'])
                    torch.nn.utils.clip_grad_norm_(disc_params, grad_norm)
                    scaler.step(optimizer['disc'])
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(disc_params, grad_norm)
                    optimizer.step()
                optimizer["disc"].zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # -----------------------
                #  Train Generator
                # -----------------------
                gen_loss = model("generator", *common_outputs)["loss"]
                gen_loss /= accum_grad_iters
            if use_amp:
                scaler.scale(gen_loss).backward()
            else:
                gen_loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                gen_params = [p for n, p in model.named_parameters() if ".generator." in n]
                if use_amp:
                    scaler.unscale_(optimizer['gen'])
                    torch.nn.utils.clip_grad_norm_(gen_params, grad_norm)
                    scaler.step(optimizer['gen'])
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(gen_params, grad_norm)
                    optimizer.step()
                optimizer["gen"].zero_grad()

            loss_dict = {"disc_loss": disc_loss * accum_grad_iters, "gen_loss": gen_loss * accum_grad_iters}

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer['gen'].param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
