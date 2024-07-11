import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import contextlib


@registry.register_model("m2m100_translator")
class M2M100Translator(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/m2m100/1_2B.yaml",
    }

    def __init__(
            self,
            model_name="facebook/m2m100_1.2B",
            max_length=200,
            lang_token_map=None,
    ):
        super().__init__()
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.lang_token_map = lang_token_map or dict()

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

        source_lang, target_lang = self.lang_token_map.get(samples["input_key"],
                                                           samples["input_key"]), self.lang_token_map.get(
            samples["output_key"], samples["output_key"])

        self.tokenizer.src_lang = source_lang

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
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                max_new_tokens=self.max_length,
                num_beams=num_beams
            )
        self.tokenizer.src_lang = target_lang
        result = self.tokenizer.batch_decode(result, skip_special_tokens=True)
        result = [_.strip() for _ in result]
        return result

    def forward(self, samples, **kwargs):

        source_texts, target_texts = samples["input"], samples["output"]
        source_lang, target_lang = self.lang_token_map.get(samples["input_key"],
                                                           samples["input_key"]), self.lang_token_map.get(
            samples["output_key"], samples["output_key"])

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        self.tokenizer.src_lang = target_lang
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
        lang_token_map = cfg.get("lang_token_map")
        model = cls(
            model_name=model_name,
            max_length=cfg.get("max_length", 200),
            lang_token_map=lang_token_map,
        )
        model.load_checkpoint_from_config(cfg)
        return model
