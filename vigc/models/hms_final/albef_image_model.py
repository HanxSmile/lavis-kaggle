import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vigc.models.blip2_models.blip2 import disabled_train
from .modules.image_hms_module import ImageHMSFeatureExtractor


@registry.register_model("albef_image_hms")
class ALBEFImageHMSClassifier(BaseModel):
    """
    copied from https://www.kaggle.com/code/crackle/efficientnetb0-pytorch-starter-lb-0-40/notebook
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/albef_image_hms_classifier.yaml",
    }

    def __init__(
            self,
            *,
            model_name="tf_efficientnet_b0",
            use_kaggle_spectrograms=True,
            use_eeg_spectrograms=False,
            freeze_encoder=False,
            use_gem=False,
            dropout=0.,
            embedding_dim=256,
            num_classes=6,
            alpha,

    ):
        super().__init__()
        self.encoder = ImageHMSFeatureExtractor(
            model_name=model_name,
            use_eeg_spectrograms=use_eeg_spectrograms,
            use_kaggle_spectrograms=use_kaggle_spectrograms,
            freeze_encoder=freeze_encoder,
            use_gem=use_gem,
            dropout=dropout,
            embedding_dim=embedding_dim
        )
        self.encoder_m = [ImageHMSFeatureExtractor(
            model_name=model_name,
            use_eeg_spectrograms=use_eeg_spectrograms,
            use_kaggle_spectrograms=use_kaggle_spectrograms,
            freeze_encoder=freeze_encoder,
            use_gem=use_gem,
            dropout=dropout,
            embedding_dim=embedding_dim
        )]
        self.class_emb = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.class_emb_m = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.logit_scale = nn.Parameter(torch.ones([])) * np.log(1 / 0.07)
        self.momentum = 0.995
        self.alpha = alpha
        self.num_classes = num_classes
        self.copy_params()
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    @torch.no_grad()
    def _momentum_update(self):
        for param, param_m in zip(self.encoder.parameters(), self.encoder_m[0].parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

        self.class_emb_m.data = self.class_emb_m.data * self.momentum + self.class_emb.data * (1. - self.momentum)

    @torch.no_grad()
    def copy_params(self):
        for param, param_m in zip(self.encoder.parameters(), self.encoder_m[0].parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.encoder_m[0].train = disabled_train

        self.class_emb_m.data.copy_(self.class_emb.data)
        self.class_emb_m.requires_grad = False

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
        label, eeg_id = samples["label"], samples["eeg_id"]
        with self.maybe_autocast():
            x_emb = self.encoder(samples)
            x_feat = x_emb / x_emb.norm(dim=1, keepdim=True)
            class_feat = self.class_emb / self.class_emb.norm(dim=1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            sim = logit_scale * x_feat @ class_feat.t()  # [B, Class]
            probs = F.softmax(sim, dim=1)  # [b, 6]

        return {"result": probs, "label": label, "eeg_id": eeg_id}

    def forward(self, samples, **kwargs):
        self.encoder_m[0] = self.encoder_m[0].to(self.device)
        with self.maybe_autocast():
            x_emb = self.encoder(samples)
            x_feat = x_emb / x_emb.norm(dim=1, keepdim=True)
            class_feat = self.class_emb / self.class_emb.norm(dim=1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            sim = logit_scale * x_feat @ class_feat.t()  # [B, Class]

            with torch.no_grad():
                self._momentum_update()
                x_emb_m = self.encoder_m[0](samples)
                x_feat_m = x_emb_m / x_emb_m.norm(dim=1, keepdim=True)
                class_feat_m = self.class_emb_m / self.class_emb_m.norm(dim=1, keepdim=True)
                sim_m = logit_scale * x_feat_m @ class_feat_m.t()
                gt = samples["label"]
                target = self.alpha * torch.softmax(sim_m, dim=1) + (1 - self.alpha) * gt

            log_prob_logits = F.log_softmax(sim, dim=1)
            loss = self.criterion(log_prob_logits, target)
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
        use_kaggle_spectrograms = cfg.get("use_kaggle_spectrograms", True)
        use_eeg_spectrograms = cfg.get("use_eeg_spectrograms", False)
        freeze_encoder = cfg.get("freeze_encoder", False)
        use_gem = cfg.get("use_gem", False)
        dropout = cfg.get("dropout", 0.)
        num_classes = cfg.get("num_classes", 6)
        alpha = cfg.get("alpha", 0.6)

        model = cls(
            model_name=model_name,
            use_kaggle_spectrograms=use_kaggle_spectrograms,
            use_eeg_spectrograms=use_eeg_spectrograms,
            freeze_encoder=freeze_encoder,
            use_gem=use_gem,
            dropout=dropout,
            num_classes=num_classes,
            alpha=alpha
        )
        model.load_checkpoint_from_config(cfg)
        return model
