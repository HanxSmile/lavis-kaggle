from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from .modules.garm import UNetGarm2DConditionModel
from .modules.vton import UNetVton2DConditionModel
import inspect
import torch
import random
import logging
from vigc.common.registry import registry
from packaging import version
from tqdm.auto import tqdm
from vigc.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import CLIPTextModel, AutoTokenizer
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@registry.register_model("ootd_viton")
class OOTDVitonNet(Blip2Base):
    """
    OOTD VITON Model
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("ootd_viton", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/vton/ootd_vton.yaml",
    }

    def __init__(
            self,
            *,
            model_name,
            vit_model_name,
            vton_model_name=None,
            garm_model_name=None,
            freeze_vton_unet=True,
            freeze_garm_unet=False,
            tokenizer_name=None,
            enable_xformers_memory_efficient_attention=False,
            gradient_checkpointing=False,
            proportion_empty_prompts=0
    ):
        super().__init__()
        assert 0 <= proportion_empty_prompts <= 1, "`--proportion_empty_prompts` must be in the range [0, 1]."
        self.proportion_empty_prompts = proportion_empty_prompts

        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet_garm = UNetGarm2DConditionModel.from_pretrained(garm_model_name or model_name, subfolder="unet")
        self.unet_vton = UNetVton2DConditionModel.from_pretrained(vton_model_name or model_name, subfolder="unet")
        if vton_model_name is None:
            self.unet_vton.replace_first_conv_layer(8)
        self.vit_image_encoder = CLIPVisionModelWithProjection.from_pretrained(vit_model_name)

        if tokenizer_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.inference_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vit_image_processor = AutoProcessor.from_pretrained(vit_model_name)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True
        )

        self.vae = self.freeze_module(self.vae)
        self.text_encoder = self.freeze_module(self.text_encoder)
        self.vit_image_encoder = self.freeze_module(self.vit_image_encoder)
        if freeze_garm_unet:
            self.unet_garm = self.freeze_module(self.unet_garm)

        if freeze_vton_unet:
            self.unet_vton = self.freeze_module(self.unet_vton)

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logging.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. \
                        If you observe problems during training, please update xFormers to at least 0.0.17. \
                        See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet_garm.enable_xformers_memory_efficient_attention()
                self.unet_vton.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if gradient_checkpointing:
            self.unet_garm.enable_gradient_checkpointing()
            self.unet_vton.enable_gradient_checkpointing()

        self._ssim_scorer = [StructuralSimilarityIndexMeasure(data_range=1.0)]
        self._lpips_scorer = [LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)]

    @property
    def ssim_scorer(self):
        self._ssim_scorer[0] = self._ssim_scorer[0].to(self.device)
        return self._ssim_scorer[0]

    @property
    def lpips_scorer(self):
        self._lpips_scorer[0] = self._lpips_scorer[0].to(self.device)
        return self._lpips_scorer[0]

    def freeze_module(self, module):
        for n, p in module.named_parameters():
            p.requires_grad = False
        module = module.eval()
        module.train = disabled_train
        logging.info(f"freezing {module.__class__.__name__}")
        return module

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

        with self.maybe_autocast():
            vton_latents = self.vae.encode(samples["vton_images"]).latent_dist.sample() * self.vae.config.scaling_factor
            garm_latents = self.vae.encode(samples["garm_images"]).latent_dist.sample() * self.vae.config.scaling_factor
            latents = self.vae.encode(samples["gt_images"]).latent_dist.sample() * self.vae.config.scaling_factor

            image_embeds = self.vit_image_encoder(samples["garm_vit_images"]).image_embeds[:, None, :]
            prompt_embeds = self.text_encoder(self.tokenize_captions(samples["captions"]), return_dict=False)[0]

            prompt_embeds[:, 1:] = image_embeds[:]

        bz = latents.shape[0]
        noise = torch.randn_like(latents)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bz,)).to(self.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with self.maybe_autocast():
            _, spatial_attn_outputs = self.unet_garm(
                garm_latents,
                0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )

            latent_vton_model_input = torch.cat([noisy_latents, vton_latents], dim=1)
            spatial_attn_inputs = spatial_attn_outputs.copy()

            # Predict the noise residual
            model_pred = self.unet_vton(
                latent_vton_model_input,
                spatial_attn_inputs,
                timesteps,
                encoder_hidden_states=prompt_embeds,
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

    def encode_prompt(
            self,
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            clip_skip=None,
    ):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if getattr(self.text_encoder.config, "use_attention_mask", False):
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask
            )
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)  # [bs, seq_len * num_img_per_prompt, dim]
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt is None:
            negative_prompt = [""] * batch_size

        if not do_classifier_free_guidance:
            return prompt_embeds, None

        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_len=prompt_embeds.shape[1],
            truncation=True,
            return_tensors="pt"
        )

        if getattr(self.text_encoder.config, "use_attention_mask", False):
            attention_mask = uncond_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_channels, height, width, generator):
        shape = (
            batch_size,
            num_channels_channels,
            int(height),
            int(width))
        latents = torch.randn(shape, generator=generator).to(self.device)
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.inference_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.inference_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def generate(
            self,
            samples,
            do_classifier_free_guidance,
            num_images_per_prompt,
            clip_skip=None,
            num_inference_steps=50,
            generator=None,
            guidance_scale=9.0,
            eta=0.0,
    ):
        prompt, negative_prompt = samples["captions"], samples.get("negative_captions", None)
        vton_image, garm_image, garm_vit_image, mask_image = samples["vton_images"], samples["garm_images"], samples[
            "garm_vit_images"], samples["mask_images"]

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
            assert len(prompt) == len(negative_prompt)
        if vton_image.ndim == 3:
            vton_image = vton_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif vton_image.ndim == 4 and vton_image.shape[0] == 1:
            vton_image = vton_image.repeat(batch_size, 1, 1, 1)

        if garm_image.ndim == 3:
            garm_image = garm_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif garm_image.ndim == 4 and garm_image.shape[0] == 1:
            garm_image = garm_image.repeat(batch_size, 1, 1, 1)

        if garm_vit_image.ndim == 3:
            garm_vit_image = garm_vit_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif garm_vit_image.ndim == 4 and garm_vit_image.shape[0] == 1:
            garm_vit_image = garm_vit_image.repeat(batch_size, 1, 1, 1)

        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif mask_image.ndim == 3:
            if mask_image.shape[0] == 1:
                mask_image = mask_image.unsqueeze(0).repeat(batch_size)
            else:
                assert mask_image.shape[0] == batch_size
                mask_image = mask_image.unsqueeze(1)
        elif mask_image.ndim == 4:
            assert mask_image.shape[0] in (1, batch_size)
            if mask_image.shape[0] == 1:
                mask_image = mask_image.repeat(batch_size, 1, 1, 1)

        assert vton_image.shape[0] == garm_image.shape[0] == garm_vit_image.shape[0] == mask_image.shape[0] == len(
            prompt)

        if negative_prompt is not None:
            assert len(negative_prompt) == batch_size

        with self.maybe_autocast():
            image_embeds = self.vit_image_encoder(garm_vit_image).image_embeds

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                clip_skip=clip_skip
            )

            image_embeds = image_embeds.repeat(1, num_images_per_prompt, 1)  # [bs, seq_len * num_img_per_prompt, dim]
            image_embeds = image_embeds.view(batch_size * num_images_per_prompt, prompt_embeds.shape[1], -1)
            prompt_embeds[:, 1:] = image_embeds[:]
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        vton_latents = self.vae.encode(vton_image).latent_dist.sample() * self.vae.config.scaling_factor
        garm_latents = self.vae.encode(garm_image).latent_dist.sample() * self.vae.config.scaling_factor
        _, _, height, width = vton_latents.shape

        vton_latents = vton_latents.repeat(1, num_images_per_prompt, 1, 1).view(batch_size * num_images_per_prompt,
                                                                                -1, height, width)
        garm_latents = garm_latents.repeat(1, num_images_per_prompt, 1, 1).view(batch_size * num_images_per_prompt,
                                                                                -1, height, width)

        if do_classifier_free_guidance:
            vton_latents = torch.cat([vton_latents] * 2)
            garm_latents = torch.cat([torch.zeros_like(garm_latents), garm_latents])

        self.inference_scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps = self.inference_scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            vton_latents.shape[1],
            height,
            width,
            generator,
        )

        noise = latents.clone()

        mask = torch.nn.functional.interpolate(
            mask_image, size=(height, width)
        )
        mask = mask.repeat(1, num_images_per_prompt, 1, 1).view(batch_size * num_images_per_prompt, -1,
                                                                int(height), int(width))

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        _, spatial_attn_outputs = self.unet_garm(
            garm_latents,
            0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            latent_vton_model_input = torch.cat([latent_model_input, vton_latents], dim=1)

            spatial_attn_inputs = spatial_attn_outputs.copy()

            # predict the noise residual
            noise_pred = self.unet_vton(
                latent_vton_model_input,
                spatial_attn_inputs,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.inference_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            init_latents_proper = vton_latents
            if i < len(timesteps) - 1:
                init_latents_proper = self.inference_scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([timesteps[i + 1]])
                )
            latents = (1 - mask) * init_latents_proper + mask * latents

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])
        return image

    @classmethod
    def from_config(cls, cfg):

        """
            model_name,
            controlnet_model_name=None,
            freeze_vae=True,
            freeze_unet=True,
            freeze_text_encoder=True,
            tokenizer_name=None,
            enable_xformers_memory_efficient_attention=False,
            gradient_checkpointing=False,
            proportion_empty_prompts=0
        """
        param_cfg = cfg.params

        model = cls(
            model_name=param_cfg.get("model_name"),
            vit_model_name=param_cfg.get("vit_model_name"),
            vton_model_name=param_cfg.get("vton_model_name", None),
            garm_model_name=param_cfg.get("garm_model_name", None),
            freeze_vton_unet=param_cfg.get("freeze_vton_unet", True),
            freeze_garm_unet=param_cfg.get("freeze_garm_unet", False),
            tokenizer_name=param_cfg.get("tokenizer_name", None),
            enable_xformers_memory_efficient_attention=param_cfg.get("enable_xformers_memory_efficient_attention", False),
            gradient_checkpointing=param_cfg.get("gradient_checkpointing", False),
            proportion_empty_prompts=param_cfg.get("proportion_empty_prompts", 0)
        )

        model.load_checkpoint_from_config(cfg)

        return model
