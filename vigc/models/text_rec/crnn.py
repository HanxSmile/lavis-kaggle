import logging
import torch
import torch.nn.functional as F
from typing import Optional
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
from .backbone.rec_mv1_enhance import MobileNetV1Enhance
from .neck.rnn import SequenceEncoder
from .head.rec_ctc_head import CTCHead
from .loss.rec_ctc_loss import CTCLoss
from .postprocess.rec_postprocess import CTCLabelDecode


@registry.register_model("mobilenetv1_enhance_crnn")
class MobilenetV1EnhanceModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/text_rec/mobilenetv1_enhance_crnn.yaml",
    }

    def __init__(
            self,
            in_channels: int = 3,
            mobilenet_scale: float = 0.5,
            neck_encoder_type: str = "rnn",
            neck_hidden_size: int = 64,
            head_mid_channels: int = 96,
            head_out_channels: int = 6625,
            character_dict_path: Optional[str] = None,
            use_space_char: bool = False,
    ):
        super().__init__()
        self.backbone = MobileNetV1Enhance(
            in_channels=in_channels, scale=mobilenet_scale)
        self.neck = SequenceEncoder(
            in_channels=self.backbone.out_channels,
            encoder_type=neck_encoder_type,
            hidden_size=neck_hidden_size
        )
        self.head = CTCHead(
            in_channels=self.neck.out_channels,
            mid_channels=head_mid_channels,
            out_channels=head_out_channels
        )
        self.criterion = CTCLoss()
        self.postprocess = CTCLabelDecode(character_dict_path=character_dict_path, use_space_char=use_space_char)

    def encode(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward(self, samples):
        images, labels, label_lengths = samples["image"], samples["label"], samples["label_length"]
        with self.maybe_autocast():
            logits = self.encode(images)
            loss = self.criterion(logits, labels, label_lengths)
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

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        images = samples["image"]
        with self.maybe_autocast():
            logits = self.encode(images)
        probs = F.softmax(logits, dim=-1)
        preds = self.postprocess(probs)
        return preds

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
        in_channels = cfg.get("in_channels", 3)
        mobilenet_scale = cfg.get("mobilenet_scale", 0.5)
        neck_encoder_type = cfg.get("neck_encoder_type", "rnn")
        neck_hidden_size = cfg.get("neck_hidden_size", 64)
        head_mid_channels = cfg.get("head_mid_channels", 96)
        head_out_channels = cfg.get("head_out_channels", 6625)
        character_dict_path = cfg.get("character_dict_path", None)
        use_space_char = cfg.get("use_space_char", False)

        model = cls(
            in_channels=in_channels,
            mobilenet_scale=mobilenet_scale,
            neck_encoder_type=neck_encoder_type,
            neck_hidden_size=neck_hidden_size,
            head_mid_channels=head_mid_channels,
            head_out_channels=head_out_channels,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
        )
        model.load_checkpoint_from_config(cfg)
        return model
