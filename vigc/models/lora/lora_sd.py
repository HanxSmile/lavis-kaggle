import logging
import random
import torch
import torch.nn.functional as F

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, \
    StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import cast_training_params
from vigc.pipelines import StableDiffusionPipeline as CustomStableDiffusionPipeline

from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib


@registry.register_model("lora_sd")
class LoraStableDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/lora/stable_diffusion.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            lora_model_name_or_path=None,
            tokenizer_name=None,
            revision=None,
            variant=None,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
            compute_dtype="fp16",
            proportion_empty_prompts=0,
            lora_rank=4,
            lora_alpha=4,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        assert compute_dtype in ["fp16", "fp32", "bf16"]
        self.compute_dtype = torch.float32
        self.proportion_empty_prompts = proportion_empty_prompts
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
        self.unet.load_adapter(lora_model_name_or_path)
        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder").to(self.compute_dtype)
        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet", prevent_training_model=False).to(self.compute_dtype)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)

        if lora_model_name_or_path is None:
            unet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)
        else:
            self.unet.load_lora_adapter(lora_model_name_or_path)
        cast_training_params(self.unet, dtype=torch.float32)
        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

    def tokenize_captions(self, examples):
        captions = []
        for caption in examples:
            if random.random() < self.proportion_empty_prompts:
                captions.append("")
            else:
                captions.append(caption)

        res = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        return res.input_ids

    def forward(self, samples):
        image, caption = samples["image"], samples["caption"]
        # Convert images to latent space
        with self.maybe_autocast(self.compute_dtype):
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        input_ids = self.tokenize_captions(caption)
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
        with self.maybe_autocast(self.compute_dtype):
            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
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
            height,
            width,
            seed=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            eta: float = 0.0,
            lora_scale: float = 1.0
    ):
        prompts = samples["caption"]
        negative_prompt = negative_prompt or samples.get("negative_prompt", None)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        pipeline = CustomStableDiffusionPipeline(
            self.unet, self.vae, self.text_encoder,
            self.inference_noise_scheduler, self.tokenizer, self.compute_dtype)
        results = pipeline.generate(
            prompts=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            eta=eta,
            cross_attention_kwargs={"scale": lora_scale}
        )
        return results

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def save_checkpoint(self, checkpoint_path):
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(self.unet)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=checkpoint_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name_or_path = cfg.get("pretrained_model_name_or_path",
                                                "stable-diffusion-v1-5/stable-diffusion-v1-5")
        lora_model_name_or_path = cfg.get("lora_model_name_or_path", None)
        tokenizer_name = cfg.get("tokenizer_name", None)
        revision = cfg.get("revision", None)
        variant = cfg.get("variant", None)
        gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        enable_xformers_memory_efficient_attention = cfg.get("enable_xformers_memory_efficient_attention", False)
        compute_dtype = cfg.get("compute_dtype", "fp16")
        proportion_empty_prompts = cfg.get("proportion_empty_prompts", 0)
        lora_rank = cfg.get("lora_rank", 4)
        lora_alpha = cfg.get("lora_alpha", 4.0)

        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            lora_model_name_or_path=lora_model_name_or_path,
            tokenizer_name=tokenizer_name,
            revision=revision,
            variant=variant,
            gradient_checkpointing=gradient_checkpointing,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            compute_dtype=compute_dtype,
            proportion_empty_prompts=proportion_empty_prompts,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        model.load_checkpoint_from_config(cfg)
        return model
