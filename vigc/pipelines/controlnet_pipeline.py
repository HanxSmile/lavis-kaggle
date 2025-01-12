import torch
from .stable_diffusion_pipeline import StableDiffusionPipeline


class ControlNetPipeline(StableDiffusionPipeline):

    def __init__(self, unet, vae, text_encoder, controlnet, scheduler, tokenizer, compute_type):
        super().__init__(unet, vae, text_encoder, scheduler, tokenizer, compute_type)
        self.controlnet = controlnet

    def generate(
            self,
            prompts,
            images,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            generator=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            eta: float = 0.0,
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

        # Prepare images
        if do_classifier_free_guidance:
            images = torch.cat([images] * 2)

        # Prepare latent variable
        height, width = images.shape[-2:]
        latents = self.prepare_latents(
            batch_size=images.shape[0],
            num_channels_channels=self.unet.config.in_channels,
            height=height // self.vae_scale_factor,
            width=width // self.vae_scale_factor,
            generator=generator
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds
            control_flag = 1 - float(
                i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
            control_scale = controlnet_conditioning_scale * control_flag
            with self.maybe_autocast(self.compute_type):
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=images,
                    conditioning_scale=control_scale,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image = self.image_processor.postprocess(image, do_denormalize=[True] * image.shape[0])
        return image
