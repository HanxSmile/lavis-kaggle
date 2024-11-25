"""Evaluation metric for Santa 2024."""

import os
from math import exp
from typing import List, Union

import transformers
import torch
import logging
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@registry.register_model("gemma2_perplexity")
class PerplexityCalculator(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/gemma/gemma2_perplexity.yaml",
    }

    PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            model_path: str = "google/gemma-2-9b",
            load_in_8bit: bool = True,
    ):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )

        self.pad_token_initialized = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info(f'Added tokenizer {self.tokenizer.pad_token}')
            self.pad_token_initialized = True

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.ignore_index = self.PAD_TOKEN_LABEL_ID

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @torch.no_grad()
    def generate(
            self, input_texts: Union[str, List[str]],
    ) -> Union[float, List[float]]:

        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []

        texts_with_special = [f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in input_texts]

        # Tokenize
        model_inputs = self.tokenizer(
            texts_with_special,
            truncation=False,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False,
        ).to(self.device)

        if 'token_type_ids' in model_inputs:
            model_inputs.pop('token_type_ids')

        # Get model output
        with self.maybe_autocast():
            output = self.model(**model_inputs, use_cache=False)
        logits = output['logits']

        if self.pad_token_initialized:
            logits = logits[:, :, :-1]

        label = model_inputs["input_ids"]

        label[label == self.tokenizer.pad_token_id] = (
            self.ignore_index  # Mask padding tokens for loss calculation
        )

        # Shift logits and labels for calculating loss
        shift_logits = logits[..., :-1, :].contiguous()  # [b, l, d]
        shift_labels = label[:, 1:].contiguous()  # [b, l]

        # Calculate token-wise loss
        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        valid_length = (shift_labels != self.ignore_index).sum(dim=-1)

        # Calculate average loss
        loss = loss.view(len(output["logits"]), -1)
        loss = torch.sum(loss, -1) / valid_length
        loss_list += loss.float().cpu().tolist()

        ppl = [exp(i) for i in loss_list]

        return ppl[0] if single_input else ppl

    @classmethod
    def from_config(cls, cfg):

        model = cls(
            model_path=cfg.get("model_path", "google/gemma-2-9b"),
            load_in_8bit=cfg.get("load_in_8bit", True),
        )

        return model
