import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import contextlib
# from vigc.models.whisper.whisper_pipeline import WhisperPipeline
from transformers import AutomaticSpeechRecognitionPipeline
from torch.nn import CrossEntropyLoss


@registry.register_model("whisper")
class Whisper(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/whisper/medium.yaml",
        "medium": "configs/models/whisper/medium.yaml",
        "small": "configs/models/whisper/small.yaml",
    }

    def __init__(
            self,
            model_name="openai/whisper-medium",
            freeze_encoder=False,
            language=None,
            task="transcribe",
            use_attention_mask=False,
    ):
        super().__init__()
        self.language = language
        self.task = task
        self.use_attention_mask = use_attention_mask
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        if freeze_encoder:
            self.model.freeze_encoder()

        if language:
            self.model.generation_config.language = language
        self.model.generation_config.task = "transcribe"

        self.model.generation_config.forced_decoder_ids = None
        if language is not None:
            self.tokenizer = WhisperTokenizer.from_pretrained(
                model_name, language=self.language, task=self.task)
            self.processor = WhisperProcessor.from_pretrained(
                model_name, language=self.language, task=self.task)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_name, language=self.language, task=self.task)
        else:
            self.tokenizer = WhisperTokenizer.from_pretrained(
                model_name, task=self.task)
            self.processor = WhisperProcessor.from_pretrained(
                model_name, task=self.task)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_name, task=self.task)

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
    def transcribe1(
            self,
            samples,
            **kwargs
    ):
        predicted_ids = self.model.generate(
            samples["input_features"],
            attention_mask=samples["attention_mask"] if self.use_attention_mask else None,
        )
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription

    @torch.no_grad()
    def transcribe2(
            self,
            samples,
            **kwargs
    ):
        inputs = samples["raw_audios"]
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.LANGUAGE, task=self.TASK)
        # ori_forced_decoder_ids = self.model.config.forced_decoder_ids
        # self.model.config.forced_decoder_ids = forced_decoder_ids
        pipe = AutomaticSpeechRecognitionPipeline(
            model=self.model,
            chunk_length_s=30,
            device=self.device,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
        )
        with self.maybe_autocast():
            generate_kwargs = {"task": "transcribe"}
            if self.language is not None:
                generate_kwargs["language"] = self.language
            transcription = pipe(
                inputs.copy(),
                batch_size=8,
                generate_kwargs=generate_kwargs,
            )
        transcription = [_["text"] for _ in transcription]
        return transcription

    @torch.no_grad()
    def generate(
            self,
            samples,
            return_loss=False,
            **kwargs
    ):
        transcription = self.transcribe2(samples)
        if not return_loss:
            return transcription

        with torch.no_grad():
            input_features = samples["input_features"]
            attention_mask = samples["attention_mask"]
            labels = samples["labels"]
            with self.maybe_autocast():
                logits = self.model(
                    input_features=input_features,
                    attention_mask=attention_mask if self.use_attention_mask else None,
                    labels=labels,
                    return_dict=True,
                ).logits
                seq_len = logits.shape[1]
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.reshape(-1))
                loss = loss.view(-1, seq_len)
                mask = (labels != -100)
                loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)

            loss = loss.tolist()
            assert len(loss) == len(transcription)
        return transcription, loss

    def forward(self, samples, **kwargs):
        input_features = samples["input_features"]
        attention_mask = samples["attention_mask"]
        labels = samples["labels"]
        with self.maybe_autocast():
            outputs = self.model(
                input_features=input_features,
                attention_mask=attention_mask if self.use_attention_mask else None,
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
        freeze_encoder = cfg.get("freeze_encoder", False)
        language = cfg.get("language", None)
        model = cls(
            model_name=model_name,
            freeze_encoder=freeze_encoder,
            language=language,
            use_attention_mask=cfg.get("use_attention_mask", False),
        )
        model.load_checkpoint_from_config(cfg)
        return model
