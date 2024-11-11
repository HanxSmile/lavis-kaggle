# TODO
import logging
import torch
import torch.nn as nn
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import torch.distributed as dist
import contextlib
from transformers import MistralModel, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
from peft import LoraConfig, get_peft_model
import random


@registry.register_model("llm_reranker")
class LLMRerankerModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bge/llm_reranker.yaml",
    }

    def __init__(
            self,
            model_name: str = "Salesforce/SFR-Embedding-2_R",
            use_grad_checkpoint: bool = False,
            query_max_len: int = 32,
            passage_max_len: int = 128,
            prompt: str = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    ):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = MistralModel.from_pretrained(
            model_name,
            quantization_config=bnb_config
        )

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

        if use_grad_checkpoint:
            self.model.gradient_checkpointing_enable()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    def last_token_pool(
            self,
            last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def forward(self, samples):
        query, pos_message, neg_message = samples['query'], samples['pos_message'], samples['neg_message']
        assert len(query) == len(pos_message) == len(neg_message)
        assert isinstance(pos_message[0], str) and isinstance(neg_message[0], list) and isinstance(query[0], str)
        group_size = len(samples["neg_message"][0])
        all_message = []
        for pos_m, neg_m_lst in zip(pos_message, neg_message):
            assert len(neg_m_lst) == group_size
            all_message.append(pos_m)
            all_message.extend(neg_m_lst)
        query = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors='pt',
        ).to(self.device)
        query = self.mask_pad_token(query)

        passage = self.tokenizer(
            all_message,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors='pt',
        ).to(self.device)
        passage = self.mask_pad_token(passage)

        with self.maybe_autocast(torch.bfloat16):
            q_reps = self.encode(query)  # [B, D]
            p_reps = self.encode(passage)  # [B * G, D]

        if self.negatives_cross_device and self.use_inbatch_neg:
            q_reps = self._dist_gather_tensor(q_reps)  # [W * B, D]
            p_reps = self._dist_gather_tensor(p_reps)  # [W * B * G, D]

        group_size = p_reps.size(0) // q_reps.size(0)  # G
        if self.use_inbatch_neg:
            scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # [W * B, W * B * G]
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)  # [1:W*B]
            target = target * group_size
            loss = self.compute_loss(scores, target)
        else:
            scores = self.compute_similarity(
                q_reps[:, None, :, ],  # [W * B, 1, D]
                p_reps.view(q_reps.size(0), group_size, -1)  # [W * B, G, D]
            ).squeeze(1) / self.temperature  # [W * B, G]

            scores = scores.view(q_reps.size(0), -1)
            target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)

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
        texts = samples['text']
        text_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max([self.query_max_len, self.passage_max_len]),
            return_tensors='pt',
        ).to(self.device)

        with self.maybe_autocast(torch.bfloat16):
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
        model_name = cfg.get("model_name", "Salesforce/SFR-Embedding-2_R")
        normalized = cfg.get("normalized", True)
        sentence_pooling_method = cfg.get("sentence_pooling_method", "cls")
        temperature = cfg.get("temperature", 0.02)
        use_inbatch_neg = cfg.get("use_inbatch_neg", False)
        negatives_cross_device = cfg.get("negatives_cross_device", False)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        query_max_len = cfg.get("query_max_len", 64)
        passage_max_len = cfg.get("passage_max_len", 256)

        model = cls(
            model_name=model_name,
            normalized=normalized,
            sentence_pooling_method=sentence_pooling_method,
            temperature=temperature,
            use_inbatch_neg=use_inbatch_neg,
            negatives_cross_device=negatives_cross_device,
            use_grad_checkpoint=use_grad_checkpoint,
            query_max_len=query_max_len,
            passage_max_len=passage_max_len,
        )
        model.load_checkpoint_from_config(cfg)
        return model
