import torch
from vigc.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline
from vigc.models.ip_adapter.attn_processor.utils import is_torch2_available

if is_torch2_available():
    from vigc.models.ip_adapter.attn_processor.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
else:
    from vigc.models.ip_adapter.attn_processor.attention_processor import IPAttnProcessor


class IPAdapterPipeline(StableDiffusionPipeline):
    def __init__(self, unet, vae, text_encoder, image_encoder, image_proj_model, scheduler, tokenizer, compute_type):
        super().__init__(unet, vae, text_encoder, scheduler, tokenizer, compute_type)
        self.image_encoder = image_encoder
        self.image_proj_model = image_proj_model

    def set_scale(self, scale=1.0):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def encode_image_prompt(
            self,
            image,
            do_classifier_free_guidance
    ):
        image_embeds = self.image_encoder(image).image_embeds
        ip_tokens = self.image_proj_model(image_embeds)
        if not do_classifier_free_guidance:
            return ip_tokens, None
        negative_ip_tokens = self.image_proj_model(torch.zeros_like(image_embeds))
        return ip_tokens, negative_ip_tokens

    def generate(
            self,
            prompts,
            prompt_images,
            height,
            width,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            generator=None,
            eta: float = 0.0,
            cross_attention_kwargs=None,
            ip_scale=1.0,
    ):
        self.set_scale(ip_scale)
        do_classifier_free_guidance = guidance_scale > 1.0
        # Prepare input prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompts,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        ip_embeds, negative_ip_embeds = self.encode_image_prompt(prompt_images, do_classifier_free_guidance)

        prompt_embeds = torch.cat([prompt_embeds, ip_embeds], dim=1)
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_ip_embeds], dim=1)
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
        self.set_scale(1.0)
        return image
