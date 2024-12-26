import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
from transformers import AutoModel, AutoTokenizer
from vigc.models.blip2_models.blip2 import disabled_train


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, similarities, soft_labels):
        similarities_prob = F.log_softmax(similarities, dim=1)
        soft_labels_prob = F.softmax(soft_labels, dim=1)
        loss = self.kl(similarities_prob, soft_labels_prob)
        return loss


@registry.register_model("bge_embedding_cls")
class BgeEmbeddingClsModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bge/bge_embedding_cls.yaml",
    }

    def __init__(
            self,
            model_name: str = "BAAI/bge-m3",
            normalized: bool = True,
            sentence_pooling_method: str = 'cls',
            query_max_len: int = 128,
            class_nums: int = 2,
            temperature: float = 1.0,
            freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.kl_loss = DistillationLoss()
        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.config = self.model.config
        self.class_embeddings = nn.Parameter(torch.randn(class_nums, self.config.hidden_size))
        self.query_max_len = query_max_len
        self.temperature = temperature
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            self.model = self.model.eval()
            self.model.train = disabled_train

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normalized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        # q_reps.shape = [B, 1, D]
        # p_reps.shape = [B, G, D]
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))  # [B, 1, G]

    def forward(self, samples):
        query, soft_labels = samples["query"], samples["soft_label"]

        query = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors='pt',
        ).to(self.device)

        with self.maybe_autocast():
            q_reps = self.encode(query)  # [B, D]
        p_reps = F.normalize(self.class_embeddings, dim=-1)

        similarity = self.compute_similarity(q_reps, p_reps)
        loss = self.kl_loss(similarity / self.temperature, soft_labels / self.temperature)

        return {"loss": loss}

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
        texts = samples['query']
        text_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors='pt',
        ).to(self.device)

        with self.maybe_autocast():
            embeddings = self.encode(text_input)  # [B, D]
        return embeddings.float()

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
        model_name = cfg.get("model_name", "BAAI/bge-m3")
        normalized = cfg.get("normalized", True)
        sentence_pooling_method = cfg.get("sentence_pooling_method", "cls")
        query_max_len = cfg.get("query_max_len", 128)
        class_nums = cfg.get("class_nums", 2)
        temperature = cfg.get("temperature", 1.0)
        freeze_backbone = cfg.get("freeze_backbone", False)

        model = cls(
            model_name=model_name,
            normalized=normalized,
            sentence_pooling_method=sentence_pooling_method,
            query_max_len=query_max_len,
            class_nums=class_nums,
            temperature=temperature,
            freeze_backbone=freeze_backbone,
        )
        model.load_checkpoint_from_config(cfg)
        return model

    def save_to_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
