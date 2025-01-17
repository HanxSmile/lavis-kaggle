import logging
import random
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, \
    T2IAdapter
from vigc.pipelines import T2IAdapterPipeline

from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
import os.path as osp


@registry.register_model("t2i_adapter_sd")
class T2IAdapterStableDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/t2i_adapter/stable_diffusion.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            adapter_model_name_or_path=None,
            tokenizer_name=None,
            revision=None,
            variant=None,
            enable_xformers_memory_efficient_attention=False,
            compute_dtype="fp16",
            proportion_empty_prompts=0,
            in_channels=3,
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        if adapter_model_name_or_path is not None:
            self.t2i_adapter = T2IAdapter.from_pretrained(adapter_model_name_or_path)
        else:
            self.t2i_adapter = T2IAdapter(
                in_channels=in_channels,
                channels=self.unet.config.block_out_channels,
                num_res_blocks=2,
                downscale_factor=self.vae_scale_factor,
                adapter_type="full_adapter",
            )
        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder").to(self.compute_dtype)
        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet").to(self.compute_dtype)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)
        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()

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
        image, condition_image, caption = samples["image"], samples["condition_image"], samples["caption"]
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
            down_block_additional_residuals = self.t2i_adapter(condition_image)
            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,
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
            seed=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            adapter_conditioning_scale=1.0,
            adapter_guidance_start=0.0,
            adapter_guidance_end=1.0,
            eta: float = 0.0,
    ):
        images = samples["condition_image"]
        prompts = samples["caption"]
        negative_prompt = negative_prompt or samples.get("negative_prompt", None)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        pipeline = T2IAdapterPipeline(self.unet, self.vae, self.text_encoder, self.t2i_adapter,
                                      self.inference_noise_scheduler, self.tokenizer, self.compute_dtype)
        results = pipeline.generate(
            prompts=prompts,
            images=images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_guidance_start=adapter_guidance_start,
            adapter_guidance_end=adapter_guidance_end,
            eta=eta,
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
        sub_dir = "t2i_adapter"
        self.t2i_adapter.save_pretrained(osp.join(checkpoint_path, sub_dir))

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name_or_path = cfg.get(
            "pretrained_model_name_or_path",
            "stable-diffusion-v1-5/stable-diffusion-v1-5")
        adapter_model_name_or_path = cfg.get("adapter_model_name_or_path", None)
        tokenizer_name = cfg.get("tokenizer_name", None)
        revision = cfg.get("revision", None)
        variant = cfg.get("variant", None)
        enable_xformers_memory_efficient_attention = cfg.get("enable_xformers_memory_efficient_attention", False)
        compute_dtype = cfg.get("compute_dtype", "fp16")
        proportion_empty_prompts = cfg.get("proportion_empty_prompts", 0)
        in_channels = cfg.get("in_channels", 3)

        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            adapter_model_name_or_path=adapter_model_name_or_path,
            tokenizer_name=tokenizer_name,
            revision=revision,
            variant=variant,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            compute_dtype=compute_dtype,
            proportion_empty_prompts=proportion_empty_prompts,
            in_channels=in_channels,
        )
        model.load_checkpoint_from_config(cfg)
        return model
