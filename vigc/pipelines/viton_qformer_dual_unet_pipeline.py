import torch
import inspect
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor


class VitonQformerDualUnetPipeline:

    def __init__(self, viton_model):
        self.viton_model = viton_model
        self.scheduler = viton_model.inference_noise_scheduler
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.viton_model.vae_scale_factor,
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

    def prepare_latents(self, batch_size, num_channels, height, width, generator):
        shape = (
            batch_size,
            num_channels,
            int(height),
            int(width))
        latents = torch.randn(shape, generator=generator).to(self.device)
        return latents * self.scheduler.init_noise_sigma

    @property
    def device(self):
        return self.viton_model.device

    @property
    def compute_dtype(self):
        return self.viton_model.compute_dtype

    def get_spatial_attn_outputs(
            self, condition_inputs, target_type, agnostic_vton_latents, do_classifier_free_guidance,
            negative_prompt, garm_masks, vae_encode_method="mode"):
        with self.viton_model.maybe_autocast(self.compute_dtype):
            condition_prompt_embeds, condition_negative_prompt_embeds = self.encode_prompt(
                condition_inputs["caption"],
                condition_inputs["instruction"],
                condition_inputs["condition_image"],
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            if do_classifier_free_guidance:
                condition_prompt_embeds = torch.cat([condition_negative_prompt_embeds, condition_prompt_embeds])
            if vae_encode_method == "mode":
                condition_latents = self.viton_model.vae.encode(condition_inputs["image"]).latent_dist.mode()
            else:
                condition_latents = self.viton_model.vae.encode(condition_inputs["image"]).latent_dist.sample()
            condition_latents = condition_latents * self.viton_model.vae.config.scaling_factor
            _, _, height, width = condition_latents.shape

        condition_noise = torch.randn_like(condition_latents)
        bsz = condition_latents.shape[0]
        # Sample a random timestep for each image
        garm_timesteps = torch.randint(0, self.viton_model.noise_scheduler.config.num_train_timesteps, (bsz,),
                                       device=self.device).long()
        zero_timesteps = torch.zeros_like(garm_timesteps).long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        condition_latents = self.viton_model.noise_scheduler.add_noise(condition_latents.float(),
                                                                       condition_noise.float(), zero_timesteps)

        if target_type == "garm":
            condition_latents = torch.cat([condition_latents, agnostic_vton_latents], dim=1)
        elif self.viton_model.cat_garm_mask:
            condition_latents = torch.cat([condition_latents, garm_masks], dim=1)

        if do_classifier_free_guidance:
            condition_model_inputs = torch.cat([condition_latents, condition_latents])
        else:
            condition_model_inputs = condition_latents
        with self.viton_model.maybe_autocast(self.compute_dtype):
            if target_type == "viton":
                _, spatial_attn_outputs = self.viton_model.garm_unet(
                    condition_model_inputs,
                    [],
                    0,
                    encoder_hidden_states=condition_prompt_embeds,
                    return_dict=False,
                )
            else:
                _, spatial_attn_outputs = self.viton_model.vton_unet(
                    condition_model_inputs,
                    [],
                    0,
                    encoder_hidden_states=condition_prompt_embeds,
                    return_dict=False,
                )
        return spatial_attn_outputs

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
            agnostic_vton_images,
            target_inputs,
            condition_inputs,
            target_type="viton",
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            generator=None,
            eta: float = 0.0,
            cross_attention_kwargs=None,
            vae_encode_method="mode"
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        # Prepare input prompts
        with self.viton_model.maybe_autocast(self.compute_dtype):
            target_prompt_embeds, target_negative_prompt_embeds = self.encode_prompt(
                target_inputs["caption"],
                target_inputs["instruction"],
                target_inputs["condition_image"],
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            if do_classifier_free_guidance:
                target_prompt_embeds = torch.cat([target_negative_prompt_embeds, target_prompt_embeds])
            if vae_encode_method == "mode":
                target_latents = self.viton_model.vae.encode(target_inputs["image"]).latent_dist.mode()
            else:
                target_latents = self.viton_model.vae.encode(target_inputs["image"]).latent_dist.sample()
            agnostic_vton_latents = self.viton_model.vae.encode(agnostic_vton_images).latent_dist.mode()
            _, channels, height, width = target_latents.shape

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        masks = torch.nn.functional.interpolate(
            target_inputs["mask"], size=(height, width)
        )
        if target_type == "garm":
            garm_masks = masks.clone()
        else:
            garm_masks = torch.nn.functional.interpolate(
                condition_inputs["mask"], size=(height, width)
            )

        latents = self.prepare_latents(
            batch_size=len(target_inputs["caption"]),
            # num_channels=self.viton_model.garm_unet.config.in_channels,
            num_channels=channels,
            height=height,
            width=width,
            generator=generator
        )
        noise = latents.clone()

        spatial_attn_outputs = self.get_spatial_attn_outputs(
            condition_inputs, target_type, agnostic_vton_latents,
            do_classifier_free_guidance, negative_prompt,
            vae_encode_method=vae_encode_method, garm_masks=garm_masks
        )

        if do_classifier_free_guidance:
            agnostic_vton_latents = torch.cat([agnostic_vton_latents, agnostic_vton_latents])
            garm_masks = torch.cat([garm_masks, garm_masks])
        if hasattr(self.viton_model, "setup_mask"):
            if target_type == "viton":
                self.viton_model.setup_mask(self.viton_model.vton_adapters, target_inputs["mask"], condition_flag=False)
                self.viton_model.setup_mask(self.viton_model.garm_adapters, condition_inputs["mask"],
                                            condition_flag=True)
            else:  # garm
                self.viton_model.setup_mask(self.viton_model.garm_adapters, target_inputs["mask"], condition_flag=False)
                self.viton_model.setup_mask(self.viton_model.vton_adapters, condition_inputs["mask"],
                                            condition_flag=True)

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            if target_type == "viton":
                latent_model_input = torch.cat([latent_model_input, agnostic_vton_latents], dim=1)
            elif self.viton_model.cat_garm_mask:
                latent_model_input = torch.cat([latent_model_input, garm_masks], dim=1)
            spatial_attn_inputs = spatial_attn_outputs.copy()
            with self.viton_model.maybe_autocast(self.compute_dtype):
                if target_type == "viton":
                    noise_pred, _ = self.viton_model.vton_unet(
                        latent_model_input,
                        spatial_attn_inputs,
                        t,
                        encoder_hidden_states=target_prompt_embeds,
                        return_dict=False,
                        cross_attention_kwargs=cross_attention_kwargs
                    )
                else:
                    noise_pred, _ = self.viton_model.garm_unet(
                        latent_model_input,
                        spatial_attn_inputs,
                        t,
                        encoder_hidden_states=target_prompt_embeds,
                        return_dict=False,
                        cross_attention_kwargs=cross_attention_kwargs
                    )
                noise_pred = noise_pred[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]
            # mask
            init_latents_proper = target_latents * self.viton_model.vae.config.scaling_factor
            if i < len(timesteps) - 1:
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([timesteps[i + 1]])
                )
            latents = (1 - masks) * init_latents_proper + masks * latents
        with self.viton_model.maybe_autocast(self.compute_dtype):
            image = self.viton_model.vae.decode(
                latents / self.viton_model.vae.config.scaling_factor,
                return_dict=False,
                generator=generator)[0]
        image = self.image_processor.postprocess(image, do_denormalize=[True] * image.shape[0])
        return image
