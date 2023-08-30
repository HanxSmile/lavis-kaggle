import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from bnunicodenormalizer import Normalizer
import contextlib
from .wav2vec_pipeline import MoEAutomaticSpeechRecognitionPipeline
import torch.nn.functional as F
import torch.nn as nn

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


@registry.register_model("bengali_moe_wav2vec")
class BengaliMoEWav2Vec(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bengali_moe_wav2vec.yaml",
    }

    def __init__(
            self,
            model_name_list=("Sameen53/cv_bn_bestModel_1",),
            processor_name="arijitx/wav2vec2-xls-r-300m-bengali",
            freeze_lm_head=False,
            post_process_flag=True,
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.model_list = nn.ModuleList([Wav2Vec2ForCTC.from_pretrained(_) for _ in model_name_list])
        self.config = self.model_list[0].config
        for model in self.model_list:
            model.config.ctc_zero_infinity = True

            for n, param in model.named_parameters():
                if "lm_head" not in n:
                    param.requires_grad = False
                elif freeze_lm_head:
                    param.requires_grad = False
        self.dropout = nn.Dropout(self.config.final_dropout)
        output_hidden_size = (
            self.config.output_hidden_size if hasattr(self.config, "add_adapter") and self.config.add_adapter else self.config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, self.config.vocab_size)
        self.weights = nn.Parameter(torch.zeros(len(model_name_list)))
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
        inputs = samples["raw_audios"]
        pipe = MoEAutomaticSpeechRecognitionPipeline(
            model=self.model_list[0],
            chunk_length_s=10,
            moe_list=self,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            decoder=self.processor.decoder,
            device=self.device
        )
        with self.maybe_autocast():
            transcription = pipe(inputs, batch_size=8)
        transcription = [_["text"] for _ in transcription]
        if self.post_process_flag:
            transcription = [postprocess(_) for _ in transcription]
        return transcription

    def extract_logits(self, input_values, attention_mask=None):
        all_hidden_states = []
        weights = F.softmax(self.weights, dim=0)
        with self.maybe_autocast():
            for i, model in enumerate(self.model_list):
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                this_weight = weights[i]
                all_hidden_states.append(outputs.hidden_states[-1] * this_weight)
            hidden_states = torch.sum(torch.stack(all_hidden_states, dim=0), dim=0)
            hidden_states = self.dropout(hidden_states)
            logits = self.lm_head(hidden_states)
        return logits

    def forward(self, samples, **kwargs):
        input_values = samples["input_values"]
        attention_mask = samples["attention_mask"]
        labels = samples["labels"]
        logits = self.extract_logits(input_values, attention_mask)
        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self.model_list[0]._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

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
        model_name_list = cfg.get("model_name_list")
        processor_name = cfg.get("processor_name")
        post_process_flag = cfg.get("post_process_flag", True)
        freeze_lm_head = cfg.get("freeze_lm_head", False)

        model = cls(
            model_name_list=model_name_list,
            processor_name=processor_name,
            freeze_lm_head=freeze_lm_head,
            post_process_flag=post_process_flag
        )
        model.load_checkpoint_from_config(cfg)
        return model
