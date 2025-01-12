import logging
import torch
from diffusers.image_processor import VaeImageProcessor
import contextlib
import inspect


class StableDiffusionPipeline:

    def __init__(self, unet, vae, text_encoder, scheduler, tokenizer, compute_type):
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.compute_type = compute_type

    @property
    def device(self):
        return list(self.unet.parameters())[0].device

    def encode_prompt(
            self,
            prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
        """

        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]

        if not do_classifier_free_guidance:
            return prompt_embeds, None

        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_channels, height, width, generator):
        shape = (
            batch_size,
            num_channels_channels,
            int(height),
            int(width))
        latents = torch.randn(shape, generator=generator).to(self.device)
        return latents * self.scheduler.init_noise_sigma

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def generate(
            self,
            prompts,
            height,
            width,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            generator=None,
            eta: float = 0.0,
            cross_attention_kwargs=None,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        # Prepare input prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompts,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size=len(prompts),
            num_channels_channels=self.unet.config.in_channels,
            height=height // self.vae_scale_factor,
            width=width // self.vae_scale_factor,
            generator=generator
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with self.maybe_autocast(self.compute_type):

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                    cross_attention_kwargs=cross_attention_kwargs
                )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image = self.image_processor.postprocess(image, do_denormalize=[True] * image.shape[0])
        return image
