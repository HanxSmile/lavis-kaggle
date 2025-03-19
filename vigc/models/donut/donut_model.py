import torch
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base
from vigc.models.donut.modeling import DonutTokenizer, DonutEncoderDecoder


@registry.register_model("donut_ocr")
class DonutOCRModel(Blip2Base):
    """
    Nougat model for Document Image 2 Markdown.
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("donut_ocr", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/vllm/donut.yaml",
    }

    def __init__(
            self,
            *,
            model_name="vikp/texify",
            max_seq_len=384,
    ):
        super().__init__()

        self.tokenizer = DonutTokenizer(model_name)

        self.model = DonutEncoderDecoder(
            model_name,
            num_tokens=len(self.tokenizer),
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.max_seq_len = max_seq_len
        self.tokenizer.max_seq_len = self.max_seq_len

    def forward(self, samples):
        image, text = samples["image"], samples["text_input"]

        text_inputs = self.tokenizer.tokenize(text).to(image.device)
        tgt_seq, tgt_mask = text_inputs["input_ids"], text_inputs["attention_mask"]
        with self.maybe_autocast():
            loss = self.model(
                pixel_values=image,
                decoder_input_ids=tgt_seq,
                decoder_attention_mask=tgt_mask
            )
        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            temperature: float = 0.2,
            do_sample: bool = False,
            top_p: float = 0.95
    ):
        image = samples["image"]
        with self.maybe_autocast():
            outputs = self.model.generate(
                pixel_values=image,
                temperature=temperature,
                max_new_tokens=self.max_seq_len,
                decoder_start_token_id=self.tokenizer.tokenizer.bos_token_id,
                decoder_end_token_id=self.tokenizer.tokenizer.eos_token_id,
                do_sample=do_sample,
                top_p=top_p
            )
        pred_tokens = self.tokenizer.detokenize(outputs)
        pred_str = self.tokenizer.token2str(outputs)
        return {"pred_tokens": pred_tokens, "pred_str": pred_str, "pred_ids": outputs}

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name", "vikp/texify")
        max_seq_len = cfg.get("max_seq_len", 384)

        model = cls(
            model_name=model_name,
            max_seq_len=max_seq_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
