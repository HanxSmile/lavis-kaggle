import logging
import torch
import torch.nn.functional as F
from typing import Optional

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler

from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib


@registry.register_model("controlnet_sd")
class ControlNetStableDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/controlnet/stable_diffusion.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            controlnet_model_name_or_path=None,
            tokenizer_name=None,
            revision=None,
            variant=None,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
            compute_dtype="fp16"
    ):
        super().__init__()
        assert compute_dtype in ["fp16", "fp32", "bf16"]
        self.compute_dtype = torch.float32
        if compute_dtype == "fp16":
            self.compute_dtype = torch.float16
        elif compute_dtype == "bf16":
            self.compute_dtype = torch.bfloat16
        if tokenizer_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=revision,
                use_fast=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, use_fast=False)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        if controlnet_model_name_or_path is not None:
            self.controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path)
        else:
            self.controlnet = ControlNetModel.from_unet(self.unet)
        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder").to(self.compute_dtype)
        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet").to(self.compute_dtype)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

    def encode(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward(self, samples):
        if self.training:
            images, labels, label_lengths = samples["image"], samples["label"], samples["label_length"]
            with self.maybe_autocast():
                logits = self.encode(images)
                loss = self.criterion(logits, labels, label_lengths)
            return {"loss": loss}
        else:
            logits = self.encode(samples)
            probs = F.softmax(logits, dim=-1)
            return probs

    def to_onnx(self, dummy_input, dynamic_axes, save_path="model.onnx"):
        input_axis_name = ['batch_size', 'channel', 'in_width', 'int_height']
        output_axis_name = ['batch_size', 'channel', 'out_width', 'out_height']
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            input_names=["input"],
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {axis: input_axis_name[axis] for axis in dynamic_axes},
                "output": {axis: output_axis_name[axis] for axis in dynamic_axes},
            },
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
        images = samples["image"]
        with self.maybe_autocast():
            logits = self.encode(images)
        probs = F.softmax(logits, dim=-1).cpu()
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
