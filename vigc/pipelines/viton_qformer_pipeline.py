import torch
import inspect
from diffusers.image_processor import VaeImageProcessor


class VitonQformerPipeline:

    def __init__(self, viton_model):
        self.viton_model = viton_model
        self.scheduler = viton_model.inference_noise_scheduler
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.viton_model.vae_scale_factor,
                                                 do_convert_rgb=True)

    def encode_prompt(
            self,
            caption,
            instruction,
            condition_image,
            do_classifier_free_guidance,
            negative_prompt=None,
    ):
        prompt_embeds = self.viton_model.get_multi_modal_embeds(
            caption,
            instruction,
            condition_image,
            training=False,
            negative_flag=False,
            negative_prompt=None,
        )
        if not do_classifier_free_guidance:
            return prompt_embeds, None

        negative_prompt_embeds = self.viton_model.get_multi_modal_embeds(
            caption,
            instruction,
            condition_image,
            training=False,
            negative_flag=True,
            negative_prompt=negative_prompt
        )
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_channels, height, width, generator):
        shape = (
            batch_size,
            num_channels_channels,
            int(height),
            int(width))
        latents = torch.randn(shape, generator=generator).to(self.viton_model.device)
        return latents * self.scheduler.init_noise_sigma

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
            instructions,
            condition_images,
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
            instructions,
            condition_images,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.viton_model.device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size=len(prompts),
            num_channels_channels=self.viton_model.unet.config.in_channels,
            height=height // self.viton_model.vae_scale_factor,
            width=width // self.viton_model.vae_scale_factor,
            generator=generator
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with self.viton_model.maybe_autocast(self.viton_model.compute_type):

                noise_pred = self.viton_model.unet(
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
        image = self.viton_model.vae.decode(latents / self.viton_model.vae.config.scaling_factor, return_dict=False,
                                            generator=generator)[0]
        image = self.viton_model.image_processor.postprocess(image, do_denormalize=[True] * image.shape[0])
        return image
