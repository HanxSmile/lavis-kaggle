import logging
import os

import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, Wav2Vec2Processor
from transformers import EncoderDecoderModel, BertTokenizer
from bnunicodenormalizer import Normalizer
import contextlib
import pyctcdecode
from vigc.models.blip2_models.blip2 import disabled_train
from normalizer import normalize

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


@registry.register_model("bengali_spelling_correction")
class BengaliSpellingCorrection(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bengali_spelling_correction.yaml",
    }

    def __init__(
            self,
            asr_model_name: str,
            asr_processor_name: str,
            bert_model_name: str = "sagorsarker/bangla-bert-base",
            post_process_flag=True,
            normalize_flag=False,
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.normalize_flag = normalize_flag
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

        for name, param in self.asr_model.named_parameters():
            param.requires_grad = False
        self.asr_model = self.asr_model.eval()
        self.asr_model.train = disabled_train
        logging.info("Freeze ASR Model.")

        self.asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        vocab_dict = self.asr_processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

        decoder = pyctcdecode.build_ctcdecoder(
            list(sorted_vocab_dict.keys()),
            os.path.join(asr_processor_name, "language_model", "5gram.bin")
        )

        self.asr_lm_processor = Wav2Vec2ProcessorWithLM(
            feature_extractor=self.asr_processor.feature_extractor,
            tokenizer=self.asr_processor.tokenizer,
            decoder=decoder
        )

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_model_name, bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def asr_predict(
            self,
            samples,
            **kwargs
    ):
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        with self.maybe_autocast():
            logits = self.asr_model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            ).logits
            y = torch.argmax(logits, dim=-1)
        y = y.detach().cpu().numpy()
        transcription = self.asr_processor.batch_decode(y, skip_special_tokens=True)

        if self.post_process_flag:
            transcription = [postprocess(_) for _ in transcription]
        if self.normalize_flag:
            transcription = [normalize(_) for _ in transcription]
        return transcription

    @torch.no_grad()
    def generate(
            self,
            samples,
            num_beams=5,
            max_length=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            temperature=1,
            **kwargs
    ):
        src_sentences = self.asr_predict(samples)
        inputs = self.tokenizer(src_sentences, return_tensors="pt", padding="longest").to(self.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty
            )
        transcription = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if self.post_process_flag:
            transcription = [postprocess(_) for _ in transcription]
        return transcription

    def forward(self, samples, **kwargs):

        src_sentences = self.asr_predict(samples)
        target_sentences = samples["sentences"]
        if self.normalize_flag:
            target_sentences = [normalize(_) for _ in target_sentences]
        inputs = self.tokenizer(src_sentences, return_tensors="pt", padding="longest").to(self.device)
        targets = self.tokenizer(target_sentences, return_tensors="pt", padding="longest").to(self.device)
        labels = targets.input_ids.masked_fill(
            targets.input_ids == self.tokenizer.pad_token_id, -100
        )
        with self.maybe_autocast():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=targets.input_ids[:, :-1].contiguous(),
                decoder_attention_mask=targets.attention_mask[:, :-1].contiguous(),
                labels=labels[:, 1:].contiguous(),
                return_dict=True,
            )
        loss = outputs.loss
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
        asr_model_name = cfg.get("asr_model_name")
        asr_processor_name = cfg.get("asr_processor_name")
        post_process_flag = cfg.get("post_process_flag", True)
        normalize_flag = cfg.get("normalize_flag", False)
        bert_model_name = cfg.get("bert_model_name")
        model = cls(
            asr_model_name=asr_model_name,
            asr_processor_name=asr_processor_name,
            bert_model_name=bert_model_name,
            post_process_flag=post_process_flag,
            normalize_flag=normalize_flag,
        )
        model.load_checkpoint_from_config(cfg)
        return model
