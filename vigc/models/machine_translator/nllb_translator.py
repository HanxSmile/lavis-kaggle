import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
import contextlib


@registry.register_model("nllb_translator")
class NLLBTranslator(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/nllb/distilled-600M.yaml",
    }

    def __init__(
            self,
            model_name="facebook/nllb-200-distilled-600M",
            source_lang="en",
            target_lang="zh",
            max_length=512,
    ):
        super().__init__()
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = NllbTokenizer.from_pretrained(model_name)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

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
            num_beams=5,
            **kwargs
    ):
        text = samples["input"]

        self.tokenizer.src_lang = self.source_lang
        self.tokenizer.tgt_lang = self.target_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with self.maybe_autocast():
            result = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_lang),
                max_new_tokens=self.max_length,
                num_beams=num_beams
            )
        result = self.tokenizer.batch_decode(result, skip_special_tokens=True)
        result = [_.strip() for _ in result]
        return result

    def forward(self, samples, **kwargs):
        source_texts, target_texts = samples["input"], samples["output"]
        self.tokenizer.src_lang = self.source_lang
        inputs = self.tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        self.tokenizer.src_lang = self.target_lang
        labels = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        labels.input_ids[labels.input_ids == self.tokenizer.pad_token_id] = -100

        with self.maybe_autocast():
            outputs = self.model(**inputs, labels=labels.input_ids)
        loss = outputs.loss
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
        source_lang = cfg.get("source_lang")
        target_lang = cfg.get("target_lang")
        model = cls(
            model_name=model_name,
            source_lang=source_lang,
            target_lang=target_lang,
            max_length=cfg.get("max_length", 512),
        )
        model.load_checkpoint_from_config(cfg)
        return model
