import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from bnunicodenormalizer import Normalizer
import contextlib

bnorm = Normalizer()


def normalize(sen):
    _words = [bnorm(word)['normalized'] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def dari(sentence):
    try:
        if sentence[-1] != "।":
            sentence += "।"
    except:
        print(sentence)
    return sentence


@registry.register_model("bengali_wav2vec")
class BengaliWav2Vec(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bengali_wav2vec.yaml",
    }

    def __init__(
            self,
            model_name="Sameen53/cv_bn_bestModel_1",
            processor_name="arijitx/wav2vec2-xls-r-300m-bengali",
            freeze_encoder=False,
            post_process_flag=True
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.config.ctc_zero_infinity = True
        if freeze_encoder:
            self.model.freeze_feature_encoder()
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_name)

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
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        with self.maybe_autocast():
            logits = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            ).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        if self.post_process_flag:
            transcription = [dari(normalize(_)) for _ in transcription]
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
        model = cls(model_name=model_name, processor_name=processor_name, freeze_encoder=freeze_encoder,
                    post_process_flag=post_process_flag)
        model.load_checkpoint_from_config(cfg)
        return model
