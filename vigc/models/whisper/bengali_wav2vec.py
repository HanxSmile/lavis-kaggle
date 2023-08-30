import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, pipeline
from bnunicodenormalizer import Normalizer
import contextlib
import random

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
            post_process_flag=True,
            # w2v_model_path="/mnt/petrelfs/hanxiao/work/bengali_utils/model/bn_w2v_model.text",
            # length_threshold=None,
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.config.ctc_zero_infinity = True
        self.model.config.ctc_loss_reduction = "sum"
        if freeze_encoder:
            self.model.freeze_feature_encoder()
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_name)
        # self.correction = BengaliSpellCorrection(w2v_model_path, length_threshold)

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
    def generate_(
            self,
            samples,
            **kwargs
    ):
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        with self.maybe_autocast():
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = outputs.logits.detach().cpu().numpy()
        transcription = []
        for l in logits:
            transcription.append(self.processor.decode(l).text)
        if self.post_process_flag:
            transcription = [postprocess(_) for _ in transcription]
        return transcription

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        inputs = samples["raw_audios"]
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            chunk_length_s=10,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            decoder=self.processor.decoder,
            device=self.device
        )
        with self.maybe_autocast():
            transcription = pipe(inputs, batch_size=8)
        transcription = [_["text"] for _ in transcription]
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

    def mixup_forward(self, samples, **kwargs):
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        labels = samples["labels"]
        bs = int(input_values.shape[0])
        bs_idx = list(range(bs))
        random.shuffle(bs_idx)
        input_values2 = input_values[bs_idx]
        attention_mask2 = attention_mask[bs_idx]
        labels2 = labels[bs_idx]

        weight = random.uniform(0, 1)
        mixup_input_values = weight * input_values + (1 - weight) * input_values2

        with self.maybe_autocast():
            outputs = self.model(
                input_values=mixup_input_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            outputs2 = self.model(
                input_values=mixup_input_values,
                attention_mask=attention_mask2,
                labels=labels2,
                return_dict=True,
            )

        loss = outputs.loss * weight + outputs2.loss * (1 - weight)
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
