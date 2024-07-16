import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import MarianMTModel, MarianConfig
from vigc.models.machine_translator.tokenization_marian import MarianTokenizer
import contextlib


@registry.register_model("opus_translator")
class OpusTranslator(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/marian/opus.yaml",
    }

    def __init__(
            self,
            model_name,
            source_lang,
            target_lang,
            source_spm,
            target_spm,
            max_length=512,
    ):
        super().__init__()
        self.config = MarianConfig.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer(
            source_spm=source_spm,
            target_spm=target_spm,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self.set_config()
        self.model = MarianMTModel(self.config)
        self.max_length = max_length

    def set_config(self):
        pad_token_id = self.tokenizer.pad_token_id
        self.config.bad_words_ids = [[pad_token_id]]
        self.config.decoder_start_token_id = pad_token_id
        self.config.decoder_vocab_size = len(self.tokenizer)
        self.config.extra_pos_embeddings = len(self.tokenizer)
        self.config.pad_token_id = pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.vocab_size = len(self.tokenizer)

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
                bos_token_id=self.tokenizer.eos_token_id,
                decoder_start_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                forced_eos_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_length,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                renormalize_logits=True
            )
        result = self.tokenizer.batch_decode(result, skip_special_tokens=True)
        result = [_.strip() for _ in result]
        return result

    def forward(self, samples, **kwargs):

        source_texts, target_texts = samples["input"], samples["output"]

        inputs = self.tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        labels = self.tokenizer(
            text_target=target_texts,
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
        source_spm = cfg.get("source_spm")
        target_spm = cfg.get("target_spm")
        model = cls(
            model_name=model_name,
            source_lang=source_lang,
            target_lang=target_lang,
            source_spm=source_spm,
            target_spm=target_spm,
            max_length=cfg.get("max_length", 512),
        )
        model.load_checkpoint_from_config(cfg)
        return model
