import logging
import torch
from vigc.models.base_model import BaseModel
from vigc.common.dist_utils import download_cached_file
from vigc.common.utils import is_url
import contextlib
import os


class GanBaseModel(BaseModel):

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            msg = self.load_state_dict(checkpoint, strict=False)

        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            msg = self.load_state_dict(checkpoint, strict=False)

        elif "generator" in url_or_filename and "discriminator" in url_or_filename:
            generator_checkpoint = torch.load(url_or_filename["generator"], map_location="cpu")
            discriminator_checkpoint = torch.load(url_or_filename["discriminator"], map_location="cpu")
            msg1 = self.generator.load_state_dict(generator_checkpoint, strict=False)
            msg2 = self.discriminator.load_state_dict(discriminator_checkpoint, strict=False)
            msg = msg1 + msg2

        else:
            raise RuntimeError("checkpoint url or path is invalid")

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info(f"Missing keys exist when loading '{url_or_filename}'.")
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def get_generator_parameter_group(self, learning_rate):
        params = [
            {"params": self.generator.parameters(), "lr": learning_rate},
        ]
        return params

    def get_discriminator_parameter_group(self, learning_rate):
        params = [
            {"params": self.discriminator.parameters(), "lr": learning_rate},
        ]
        return params

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_pretrained = cfg.get("load_pretrained", False)
        if load_pretrained:
            pretrained_path = cfg.get("pretrained", None)
            assert pretrained_path is not None, "Found load_pretrained is True, but pretrained_path is None"
            self.load_checkpoint(url_or_filename=pretrained_path)
            logging.info(f"Loaded pretrained checkpoint from {pretrained_path}")

        load_finetuned = cfg.get("load_finetuned", False)

        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info(f"Loaded finetuned model '{finetune_path}'.")
