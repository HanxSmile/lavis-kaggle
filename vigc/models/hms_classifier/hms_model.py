import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from vigc.models.blip2_models.blip2 import disabled_train
import contextlib
import timm
import torch.nn as nn
from vigc.models.hms_classifier.modules import GeM, KLDivLossWithLogits
import torch.nn.functional as F


@registry.register_model("hms_classifier")
class HMSClassifier(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/hms_classifier.yaml",
    }

    def __init__(
            self,
            model_name="tf_efficientnet_b0",
            input_channels=32,
            freeze_encoder=False,
            separate_head=False
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=input_channels)
        num_in_features = self.backbone.get_classifier().in_features
        if freeze_encoder:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
            self.backbone = self.backbone.eval()
            self.backbone.train = disabled_train
            logging.warning("Freeze the backbone model")

        self.separate_head = separate_head
        if separate_head:
            self.head = nn.ModuleList(
                [
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                    nn.Sequential(GeM(), nn.Linear(num_in_features, 1)),
                ]
            )
        else:
            self.head = nn.Sequential(GeM(), nn.Linear(num_in_features, 6))
        self.criterion = KLDivLossWithLogits()

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
        eeg_image, spec_image, label = samples["eeg_image"], samples["spec_image"], samples["label"]
        eeg_id, spec_id, uid = samples["eeg_id"], samples["spec_id"], samples["uid"]
        model_inputs = torch.cat([eeg_image] + [spec_image] * 16, dim=1)
        outputs = []
        with self.maybe_autocast():
            features = self.backbone.forward_features(model_inputs)
            for head in self.head:
                outputs.append(head(features))
            logits = torch.cat(outputs, dim=-1)  # [b, 6]
            probs = F.softmax(logits, dim=1)  # [b, 6]
            # loss = self.eval_criterion(logits, label).sum(dim=1)  # [b, 1]

        return {"logits": logits, "probs": probs, "label": label, "eeg_id": eeg_id, "spec_id": spec_id, "uid": uid}

    def forward(self, samples, **kwargs):
        eeg_image, spec_image, label = samples["eeg_image"], samples["spec_image"], samples["label"]
        model_inputs = torch.cat([eeg_image] + [spec_image] * 16, dim=1)

        with self.maybe_autocast():
            features = self.backbone.forward_features(model_inputs)
            if self.separate_head:
                outputs = []
                for head in self.head:
                    outputs.append(head(features))
                logits = torch.cat(outputs, dim=-1)  # [b, 6]
            else:
                logits = self.head(features)  # [b, 6]
            loss = self.criterion(logits, label)
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
        input_channels = cfg.get("input_channels")
        freeze_encoder = cfg.get("freeze_encoder", 32)
        separate_head = cfg.get("separate_head", False)

        model = cls(
            model_name=model_name,
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            separate_head=separate_head
        )
        model.load_checkpoint_from_config(cfg)
        return model
