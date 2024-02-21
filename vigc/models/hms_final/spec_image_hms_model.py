import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from vigc.models.blip2_models.blip2 import disabled_train
import contextlib
import timm
import torch.nn as nn
from vigc.models.hms_classifier.modules import GeM
import torch.nn.functional as F


@registry.register_model("spec_image_hms")
class SpecImageHMSClassifier(BaseModel):
    """
    copied from https://www.kaggle.com/code/yunsuxiaozi/hms-baseline-resnet34d-512-512-training
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/spec_image_hms_classifier.yaml",
    }

    def __init__(
            self,
            model_name="tf_efficientnet_b0",
            freeze_encoder=False,
            use_gem=False,
            dropout=0.
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=1)
        num_in_features = self.backbone.get_classifier().in_features
        if freeze_encoder:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
            self.backbone = self.backbone.eval()
            self.backbone.train = disabled_train
            logging.warning("Freeze the backbone model")

        if use_gem:
            self.head = nn.Sequential(
                GeM(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(num_in_features, 6)
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(num_in_features, 6)
            )
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

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
        model_inputs, label, eeg_id = samples["image"], samples["label"], samples["eeg_id"]
        with self.maybe_autocast():
            features = self.backbone.forward_features(model_inputs)
            logits = self.head(features)
            probs = F.softmax(logits, dim=1)  # [b, 6]
            # loss = self.eval_criterion(logits, label).sum(dim=1)  # [b, 1]

        return {"result": probs, "label": label, "eeg_id": eeg_id}

    def forward(self, samples, **kwargs):
        model_inputs, label = samples["image"], samples["label"]
        with self.maybe_autocast():
            features = self.backbone.forward_features(model_inputs)
            logits = self.head(features)
            log_prob_logits = F.log_softmax(logits, dim=1)
            loss = self.kl_loss(log_prob_logits, label)
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
        freeze_encoder = cfg.get("freeze_encoder", False)
        use_gem = cfg.get("use_gem", False)
        dropout = cfg.get("dropout", 0.)

        model = cls(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            use_gem=use_gem,
            dropout=dropout
        )
        model.load_checkpoint_from_config(cfg)
        return model
