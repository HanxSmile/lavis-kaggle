import logging
import torch
import torch.nn as nn
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
import torch.nn.functional as F
import timm


@registry.register_model("timm_image_regression")
class TimmImageRegressor(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/timm_image_regression.yaml",
    }

    def __init__(
            self,
            model_name="./mobilenetv3_large_100.pth",
            label_expand_ratio=1.0
    ):
        super().__init__()
        self.model = self.prepare_model(model_name)
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_expand_ratio = label_expand_ratio

    def prepare_model(self, model_name):
        model_arch = model_name.split("/")[-1].split(".")[0]
        backbone = timm.create_model(model_arch, pretrained=False)
        ckpt = torch.load(model_name, map_location=torch.device('cpu'))
        backbone.load_state_dict(ckpt)
        n_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        head = nn.Linear(n_features, 1)
        return nn.Sequential(backbone, head)

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", False)

        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info(f"Loaded finetuned model '{finetune_path}'.")

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        with self.maybe_autocast():
            logits = self.model(samples["image"]).squeeze(-1)
            probs = F.softmax(logits, dim=-1) / self.label_expand_ratio
        return probs

    def forward(self, samples, **kwargs):
        with self.maybe_autocast():
            logits = self.model(samples["image"]).squeeze(-1)
            loss = self.criterion(logits, samples["label"] * self.label_expand_ratio)
        return {"loss": loss}

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name")
        label_expand_ratio = cfg.get("label_expand_ratio", 1.0)

        model = cls(
            model_name=model_name,
            label_expand_ratio=label_expand_ratio
        )
        model.load_checkpoint_from_config(cfg)
        return model
