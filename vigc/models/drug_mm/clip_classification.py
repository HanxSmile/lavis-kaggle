import logging
import torch
import torch.nn as nn
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from vigc.models.blip2_models.blip2 import disabled_train
import contextlib
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, logits, attention_mask=None):
        # last_hidden_state.shape = [B, L, C]
        # attention_mask.shape = [B, L]
        last_hidden_state = logits
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        else:
            input_mask_expanded = torch.ones_like(last_hidden_state)

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


@registry.register_model("clip_classification")
class ClipClassifier(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/clip_classification.yaml",
    }

    def __init__(
            self,
            model_name="openai/clip-vit-base-patch32",
            text_model=None,
            num_classes=1,
            dropout=0.2,
            freeze_text_encoder=True,
            freeze_image_encoder=True,
            mean_pooling=True,
    ):
        super().__init__()
        self.mean_pooling = mean_pooling
        self.use_bert = text_model is not None
        self.visual_model = CLIPVisionModel.from_pretrained(model_name)
        if not self.use_bert:
            self.textual_model = CLIPTextModel.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(text_model)
            config.hidden_dropout = 0.
            config.hidden_dropout_prob = 0.
            config.attention_dropout = 0.
            config.attention_prob_dropout_prob = 0.
            self.textual_model = AutoModel.from_pretrained(text_model, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(text_model)

        if freeze_text_encoder:
            for n, p in self.textual_model.named_parameters():
                p.requires_grad = False

            self.textual_model.eval()
            self.textual_model.train = disabled_train
            logging.info("The text model has been frozen.")

        if freeze_image_encoder:
            for n, p in self.visual_model.named_parameters():
                p.requires_grad = False
            self.visual_model.eval()
            self.visual_model.train = disabled_train
            logging.info("The image model has been frozen.")

        self.pooler = MeanPooling() if mean_pooling else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.textual_model.config.hidden_size + self.visual_model.config.hidden_size,
                      self.visual_model.config.hidden_size),
            nn.BatchNorm1d(num_features=self.visual_model.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.visual_model.config.hidden_size, num_classes)
        )
        self.criterion = nn.BCEWithLogitsLoss()

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
            input_ids, attention_mask = self.preprocess_inputs(samples)
            if self.mean_pooling:
                text_features = self.textual_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                image_features = self.visual_model(pixel_values=samples["image"]).last_hidden_state

                text_feature = self.pooler(text_features, attention_mask)
                image_feature = self.pooler(image_features)
            else:
                text_feature = self.textual_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
                image_feature = self.visual_model(pixel_values=samples["image"]).pooler_output
            total_feature = torch.cat([text_feature, image_feature], dim=1)
            logits = self.head(total_feature)
            probs = F.sigmoid(logits).squeeze(-1)
        return {"result": probs, "label": samples["label"], "id": samples["id"],
                "label_index": samples.get("label_index", None)}

    def preprocess_inputs(self, x):
        if not self.use_bert:
            text_inputs = self.tokenizer(
                x["text"],
                padding='longest',
                truncation=True,
                max_length=77,
                # padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            text_inputs = self.tokenizer(
                x["text"],
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
        return text_inputs.input_ids, text_inputs.attention_mask

    def forward(self, samples, **kwargs):
        with self.maybe_autocast():
            input_ids, attention_mask = self.preprocess_inputs(samples)
            if self.mean_pooling:
                text_features = self.textual_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                image_features = self.visual_model(pixel_values=samples["image"]).last_hidden_state

                text_feature = self.pooler(text_features, attention_mask)
                image_feature = self.pooler(image_features)
            else:
                text_feature = self.textual_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
                image_feature = self.visual_model(pixel_values=samples["image"]).pooler_output
            total_feature = torch.cat([text_feature, image_feature], dim=1)
            logits = self.head(total_feature).squeeze(-1)
            loss = self.criterion(logits, samples["label"])
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
        num_classes = cfg.get("num_classes")
        dropout = cfg.get("dropout", 0.2)
        freeze_text_encoder = cfg.get("freeze_text_encoder", True)
        freeze_image_encoder = cfg.get("freeze_image_encoder", True)
        mean_pooling = cfg.get("mean_pooling", True)
        text_model = cfg.get("text_model", None)

        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_image_encoder=freeze_image_encoder,
            freeze_text_encoder=freeze_text_encoder,
            mean_pooling=mean_pooling,
            text_model=text_model
        )
        model.load_checkpoint_from_config(cfg)
        return model
