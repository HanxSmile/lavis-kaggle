import logging
import torch
import torch.nn as nn
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from vigc.models.blip2_models.blip2 import disabled_train
import contextlib
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, logits, attention_mask):
        last_hidden_state = logits.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


@registry.register_model("bert_classification")
class BertClassifier(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bert_classification.yaml",
    }

    def __init__(
            self,
            model_name="google-bert/bert-base-multilingual-cased",
            num_classes=1,
            dropout=0.2,
            freeze_encoder=False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_prob_dropout_prob = 0.
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        if freeze_encoder:
            for n, p in self.model.names_parameters():
                p.requires_grad = False
            self.model = self.model.eval()
            self.model.train = disabled_train
            logging.warning("Freeze the backbone model")

        self.pooler = MeanPooling()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, num_classes)
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
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            out = self.pooler(out, attention_mask)
            logits = self.head(out)
            probs = F.sigmoid(logits)
        return {"result": probs, "label": samples["label"], "id": samples["id"], "row": samples["row"],
                "text": samples["text"]}

    def preprocess_inputs(self, x):
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
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            out = self.pooler(out, attention_mask)
            logits = self.head(out)
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
        freeze_encoder = cfg.get("freeze_encoder", False)

        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            dropout=dropout
        )
        model.load_checkpoint_from_config(cfg)
        return model
