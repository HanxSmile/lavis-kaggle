import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from bnunicodenormalizer import Normalizer
import contextlib
from transformers import pipeline

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


@registry.register_model("bengali_whisper")
class BengaliWhisper(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bengali_whisper_medium.yaml",
        "medium": "configs/models/bengali_whisper_medium.yaml",
        "small": "configs/models/bengali_whisper_small.yaml",
    }

    LANGUAGE = "bn"
    TASK = "transcribe"

    def __init__(
            self,
            model_name="openai/whisper-medium",
            freeze_encoder=False,
            post_process_flag=True
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        if freeze_encoder:
            self.model.freeze_encoder()
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=self.LANGUAGE, task=self.TASK)
        self.processor = WhisperProcessor.from_pretrained(model_name, language=self.LANGUAGE, task=self.TASK)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

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
        inputs = samples["raw_audios"]
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.LANGUAGE, task=self.TASK)
        ori_forced_decoder_ids = self.model.config.forced_decoder_ids
        self.model.config.forced_decoder_ids = forced_decoder_ids
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            chunk_length_s=8,
            device=self.device,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor
        )
        with self.maybe_autocast():
            transcription = pipe(inputs.copy(), batch_size=8)["text"]

        if self.post_process_flag:
            transcription = [dari(normalize(_)) for _ in transcription]
        self.model.config.forced_decoder_ids = ori_forced_decoder_ids
        return transcription

    def forward(self, samples, **kwargs):
        input_features = samples["input_features"]
        labels = samples["labels"]
        with self.maybe_autocast():
            outputs = self.model(
                input_features=input_features,
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
        post_process_flag = cfg.get("post_process_flag", True)
        freeze_encoder = cfg.get("freeze_encoder", False)
        model = cls(model_name=model_name, freeze_encoder=freeze_encoder, post_process_flag=post_process_flag)
        model.load_checkpoint_from_config(cfg)
        return model
