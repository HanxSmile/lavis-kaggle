import logging
import random
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler
from diffusers.training_utils import cast_training_params
from vigc.pipelines import IPAdapterPipeline

from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib

from vigc.models.ip_adapter.attn_processor.ip_adapter import ImageProjModel
from vigc.models.ip_adapter.attn_processor.utils import is_torch2_available

if is_torch2_available():
    from vigc.models.ip_adapter.attn_processor.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, \
        AttnProcessor2_0 as AttnProcessor
else:
    from vigc.models.ip_adapter.attn_processor.attention_processor import IPAttnProcessor, AttnProcessor


@registry.register_model("ip_adapter_sd")
class IPAdapterStableDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/ip_adapter/stable_diffusion.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            image_encoder_path,
            tokenizer_name=None,
            revision=None,
            variant=None,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
            compute_dtype="fp16",
            proportion_empty_prompts=0,
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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
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
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder").to(self.compute_dtype)
        self.image_encoder = self.freeze_module(self.image_encoder, "image_encoder").to(self.compute_dtype)
        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet", prevent_training_model=False).to(self.compute_dtype)
        self.adapter_modules = self.register_attn_processors(self.unet)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)

        cast_training_params(self.unet, dtype=torch.float32)
        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

    def register_attn_processors(self, unet):
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        return adapter_modules

    def get_encoding_embeddings(self, captions, clip_images):
        image_embeds = self.image_encoder(clip_images).image_embeds
        prompts = []
        clip_encodings = []
        for caption, image_embed in zip(captions, image_embeds):
            random_num = random.random()
            this_caption = caption
            this_image_encoding = image_embed
            if random_num < self.proportion_empty_prompts:
                this_image_encoding = torch.zeros_like(image_embed)
            elif random_num < 2 * self.proportion_empty_prompts:
                this_caption = ""
            elif random_num < 3 * self.proportion_empty_prompts:
                this_caption = ""
                this_image_encoding = torch.zeros_like(image_embed)
            prompts.append(this_caption)
            clip_encodings.append(this_image_encoding)
        clip_encodings = torch.stack(clip_encodings, dim=0)
        text_input_ids = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device).input_ids

        text_encodings = self.text_encoder(text_input_ids, return_dict=False)[0]
        ip_tokens = self.image_proj_model(clip_encodings)
        encoder_hidden_states = torch.cat([text_encodings, ip_tokens], dim=1)
        return encoder_hidden_states

    def forward(self, samples):
        image, clip_image, caption = samples["image"], samples["condition_image"], samples["caption"]
        # Convert images to latent space
        with self.maybe_autocast(self.compute_dtype):
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self.get_encoding_embeddings(caption, clip_image)

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

            state_dict = torch.load(finetune_path, map_location="cpu")

            # Load state dict for image_proj_model and adapter_modules
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

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
            ip_scale: float = 1.0
    ):
        prompts = samples["caption"]
        negative_prompt = negative_prompt or samples.get("negative_prompt", None)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        pipeline = IPAdapterPipeline(
            self.unet, self.vae, self.text_encoder, self.image_encoder, self.image_proj_model,
            self.inference_noise_scheduler, self.tokenizer, self.compute_dtype)
        results = pipeline.generate(
            prompts=prompts,
            prompt_images=samples["condition_image"],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            eta=eta,
            ip_scale=ip_scale
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
        ckpt = {
            "image_proj": self.image_proj_model.state_dict(),
            "ip_adapter": self.adapter_modules.state_dict(),
        }

        torch.save(ckpt, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name_or_path = cfg.get("pretrained_model_name_or_path",
                                                "stable-diffusion-v1-5/stable-diffusion-v1-5")
        image_encoder_path = cfg.get("image_encoder_path", None)
        tokenizer_name = cfg.get("tokenizer_name", None)
        revision = cfg.get("revision", None)
        variant = cfg.get("variant", None)
        gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        enable_xformers_memory_efficient_attention = cfg.get("enable_xformers_memory_efficient_attention", False)
        compute_dtype = cfg.get("compute_dtype", "fp16")
        proportion_empty_prompts = cfg.get("proportion_empty_prompts", 0)

        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            image_encoder_path=image_encoder_path,
            tokenizer_name=tokenizer_name,
            revision=revision,
            variant=variant,
            gradient_checkpointing=gradient_checkpointing,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            compute_dtype=compute_dtype,
            proportion_empty_prompts=proportion_empty_prompts,
        )
        model.load_checkpoint_from_config(cfg)
        return model
