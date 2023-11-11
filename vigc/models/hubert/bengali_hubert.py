import logging
import os

import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import HubertForCTC, Wav2Vec2Processor
from bnunicodenormalizer import Normalizer
import contextlib
import pyctcdecode

bnorm = Normalizer()


def postprocess(sentence):
    period_set = [".", "?", "!", "।"]
    _words = [bnorm(word)['normalized'] for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    try:
        if sentence[-1] not in period_set:
            sentence += "।"
    except:
        # print(sentence)
        sentence = "।"
    return sentence


@registry.register_model("bengali_hubert_xlarge")
class BengaliHubert(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bengali_hubert_xlarge.yaml",
    }

    def __init__(
            self,
            model_name="facebook/hubert-xlarge-ll60k",
            processor_name="arijitx/wav2vec2-xls-r-300m-bengali",
            loss_reduction="mean",
            freeze_encoder=False,
            post_process_flag=True
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(
            model_name,
            activation_dropout=0.05,
            final_dropout=0.05,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.0,
            ctc_loss_reduction=loss_reduction,
            ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
        if freeze_encoder:
            self.model.freeze_feature_encoder()
        vocab_dict = processor.tokenizer.get_vocab()["ben"]
        vocab_dict['<s>'] = 64
        vocab_dict['</s>'] = 65
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

        self.decoder = pyctcdecode.build_ctcdecoder(
            list(sorted_vocab_dict.keys()),
            os.path.join(processor_name, "language_model", "5gram.bin")
        )

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
        transcription = []
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        with self.maybe_autocast():
            logits = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            ).logits
        logits = logits.detach().cpu().numpy()
        for l in logits:
            sentence = self.decoder.decode_beams(l, beam_width=512)[0][0]
            transcription.append(sentence)
        if self.post_process_flag:
            transcription = [postprocess(_) for _ in transcription]
        return transcription

    def forward(self, samples, **kwargs):
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        labels = samples["labels"]
        with self.maybe_autocast():
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
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
        processor_name = cfg.get("processor_name")
        post_process_flag = cfg.get("post_process_flag", True)
        freeze_encoder = cfg.get("freeze_encoder", False)
        loss_reduction = cfg.get("loss_reduction", "mean")
        model = cls(model_name=model_name, processor_name=processor_name,
                    freeze_encoder=freeze_encoder, post_process_flag=post_process_flag, loss_reduction=loss_reduction)
        model.load_checkpoint_from_config(cfg)
        return model