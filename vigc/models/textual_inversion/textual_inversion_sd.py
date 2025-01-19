import random
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler
from vigc.pipelines import StableDiffusionPipeline

from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
import os.path as osp


@registry.register_model("textual_inversion_sd")
class TextualInversionStableDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/textual_inversion/stable_diffusion.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            tokenizer_name=None,
            revision=None,
            variant=None,
            enable_xformers_memory_efficient_attention=False,
            gradient_checkpointing=False,
            compute_dtype="fp16",
            proportion_empty_prompts=0,
            initializer_token=None,
            placeholder_token=None,
    ):
        super().__init__()
        if isinstance(placeholder_token, str):
            self.placeholder_tokens = [placeholder_token]
        if isinstance(initializer_token, str):
            self.initializer_tokens = [initializer_token]
        if len(self.initializer_tokens) == 1:
            self.initializer_tokens = self.initializer_tokens * len(self.placeholder_tokens)
        assert len(self.initializer_tokens) == len(self.placeholder_tokens)

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

        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        if num_added_tokens != len(self.placeholder_tokens):
            raise ValueError(
                f"The tokenizer already contains the token {self.placeholder_tokens}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        # Convert the initializer_token, placeholder_token to ids
        initializer_token_ids = []
        for initializer_token in self.initializer_tokens:
            token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")
            initializer_token_id = token_ids[0]
            initializer_token_ids.append(initializer_token_id)

        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id, initializer_token_id in zip(self.placeholder_token_ids, initializer_token_ids):
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()
        self.orig_embeds_params = token_embeds.clone().detach()
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet", prevent_training_model=False).to(self.compute_dtype)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)
        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

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

    def prevent_embeds_update(self):
        # Let's make sure we don't update any embedding weights besides the newly added token
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[min(self.placeholder_token_ids): max(self.placeholder_token_ids) + 1] = False

        with torch.no_grad():
            self.text_encoder.get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[
                index_no_updates]

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
        self.prevent_embeds_update()

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
        if not load_finetuned:
            return

        finetune_path = cfg.get("finetuned", None)
        assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
        if isinstance(finetune_path, str):
            finetune_path = [finetune_path]
        ckpt = {}
        for p in finetune_path:
            ckpt.update(torch.load(p, map_location=self.device))
        for placeholder_token in ckpt.keys():
            assert placeholder_token in self.placeholder_tokens

        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id, placeholder_token in zip(self.placeholder_token_ids, self.placeholder_tokens):
                token_embeds[token_id] = ckpt[placeholder_token].clone()
        print("Loaded finetuned model")

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
    ):
        prompts = samples["caption"]
        negative_prompt = negative_prompt or samples.get("negative_prompt", None)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        self.prevent_embeds_update()
        pipeline = StableDiffusionPipeline(self.unet, self.vae, self.text_encoder,
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
        learned_embeds = (
            self.text_encoder.get_input_embeddings().weight[
            min(self.placeholder_token_ids): max(self.placeholder_token_ids) + 1]
        )
        learned_embeds_dict = {placeholder_token: learned_embeds[i].detach().cpu() for i, placeholder_token in
                               enumerate(self.placeholder_tokens)}
        save_path = osp.join(checkpoint_path, "learned_embeds.bin")
        torch.save(learned_embeds_dict, save_path)
        print("Saved learned_embeds.bin to {}".format(save_path))

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name_or_path = cfg.get(
            "pretrained_model_name_or_path",
            "stable-diffusion-v1-5/stable-diffusion-v1-5")
        tokenizer_name = cfg.get("tokenizer_name", None)
        revision = cfg.get("revision", None)
        variant = cfg.get("variant", None)
        enable_xformers_memory_efficient_attention = cfg.get("enable_xformers_memory_efficient_attention", False)
        compute_dtype = cfg.get("compute_dtype", "fp16")
        proportion_empty_prompts = cfg.get("proportion_empty_prompts", 0)
        initializer_token = cfg.get("initializer_token", None)
        placeholder_token = cfg.get("placeholder_token", None)

        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_name=tokenizer_name,
            revision=revision,
            variant=variant,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            compute_dtype=compute_dtype,
            proportion_empty_prompts=proportion_empty_prompts,
            initializer_token=initializer_token,
            placeholder_token=placeholder_token,
        )
        model.load_checkpoint_from_config(cfg)
        return model
