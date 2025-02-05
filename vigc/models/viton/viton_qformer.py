import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from typing import Literal
from transformers import AutoTokenizer
from vigc.models.viton.modules.clip_text_model import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler
from vigc.pipelines import VitonQformerPipeline
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base
import contextlib


@registry.register_model("viton_qformer")
class VitonQformer(Blip2Base):
    """
    Viton Qformer model
    Qformer is finetuned using this model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/viton/viton_qformer.yaml",
    }

    def __init__(
            self,
            *,
            pretrained_model_name_or_path,
            tokenizer_name=None,
            revision=None,
            variant=None,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
            compute_dtype="fp16",
            proportion_empty_prompts=0,
            lora_config=None,
            # q-former
            num_query_token=32,
            max_txt_len=128,
            qformer_model_name_or_path="bert-base-uncased",
            # q-former visual encoder
            vit_model="eva_clip_g",
            vit_model_ckpt=None,
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_qformer=False,
            freeze_text_encoder=False,
            freeze_vit=True,
            freeze_vit_ln=False,
            condition_image: Literal['garm', 'viton', 'random'] = "random",
            target_image: Literal['garm', 'viton', 'random'] = "random",
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # assert (not freeze_qformer) and (not freeze_text_encoder)
        assert compute_dtype in ["fp16", "fp32", "bf16"]
        self.compute_dtype = torch.float32
        self.condition_image = condition_image
        self.target_image = target_image
        self.proportion_empty_prompts = proportion_empty_prompts
        self.max_txt_len = max_txt_len
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                revision=revision,
                use_fast=False
            )
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
        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder", prevent_training_model=False).to(
            self.compute_dtype)

        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)
        self.unet = self.freeze_module(self.unet, "unet", prevent_training_model=False).to(self.compute_dtype)

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)

        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Q-former
        self.qformer_tokenizer = self.init_tokenizer(truncation_side="left", tokenizer_name=qformer_model_name_or_path)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, cached_file=vit_model_ckpt
        )
        if freeze_vit:
            self.visual_encoder = self.freeze_module(self.visual_encoder, "visual_encoder").to(self.compute_dtype)
        if freeze_vit_ln:
            self.ln_vision = self.freeze_module(self.ln_vision, "ln_vision").to(self.compute_dtype)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, model_name_or_path=qformer_model_name_or_path
        )
        self.Qformer.resize_token_embeddings(len(self.qformer_tokenizer))
        self.Qformer.cls = None

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.text_encoder.config.hidden_size
        )
        # text encoder
        lora_config = LoraConfig(
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias="none",  # won't use bias currently
            modules_to_save=[],  # TODO: might be helpful if save partial model
            # task_type="FEATURE_EXTRACTION",
        )
        self.text_encoder = get_peft_model(self.text_encoder, peft_config=lora_config)
        self.text_encoder.print_trainable_parameters()
        logging.info("Loading Lora done.")

        if freeze_qformer:
            self.Qformer = self.freeze_module(self.Qformer, "Qformer")
            self.query_tokens.requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder")

    def q_former_embeds(self, image, text):
        with self.maybe_autocast(self.compute_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.qformer_tokenizer(
            text,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast(self.compute_dtype):
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embedding = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        return image_embedding

    def get_multi_modal_embeds(self, caption, instruction, image, training=True, negative_flag=False,
                               negative_prompt=None):
        if training:
            assert not negative_flag
        with self.maybe_autocast(self.compute_dtype):
            image_embed = self.ln_vision(self.visual_encoder(image))
        captions = []
        instructions = []
        image_embeds = []
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(caption)
        else:
            negative_prompt = [""] * len(caption)
        for i, (c, ins) in enumerate(zip(caption, instruction)):
            random_num = random.random()
            this_caption = c
            this_ins = ins
            this_image_embed = image_embed[i]
            if training:
                if random_num < self.proportion_empty_prompts:
                    this_caption = ""
                elif random_num < 2 * self.proportion_empty_prompts:
                    this_ins = ""
                    this_image_embed = torch.zeros_like(this_image_embed)
                elif random_num < 3 * self.proportion_empty_prompts:
                    this_caption = ""
                    this_ins = ""
                    this_image_embed = torch.zeros_like(this_image_embed)
            elif negative_flag:
                this_caption = negative_prompt[i]
                this_ins = ""
                this_image_embed = torch.zeros_like(this_image_embed)
            captions.append(this_caption)
            instructions.append(this_ins)
            image_embeds.append(this_image_embed)
        image_embeds = torch.stack(image_embeds, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.qformer_tokenizer(
            instructions,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast(self.compute_dtype):
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embedding = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

        caption_input_ids = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device).input_ids
        text_hidden_states = self.text_encoder.get_base_model()(caption_input_ids, return_dict=False)[0]
        image_hidden_states = self.text_encoder(input_embeds=image_embedding, return_dict=False)[0]
        multi_modal_hidden_states = torch.cat([text_hidden_states, image_hidden_states], dim=1)
        return multi_modal_hidden_states

    def tokenize_captions(self, captions):
        res = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        return res.input_ids

    def forward(self, samples):
        samples = self.prepare_inputs(samples)
        image, condition_image, caption, instruction = (samples["image"], samples["condition_image"],
                                                        samples["caption"], samples["instruction"])
        # Convert images to latent space
        with self.maybe_autocast(self.compute_dtype):
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
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
        encoder_hidden_states = self.get_multi_modal_embeds(caption, instruction, condition_image, training=True)
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
            condition_image="garm",
            target_image="viton",
            seed=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=None,
            eta: float = 0.0,
    ):
        samples = self.prepare_inputs(samples, condition_image=condition_image, target_image=target_image)
        prompts = samples["caption"]
        images = samples["image"]
        masks = samples["mask"]
        instructions = samples["instruction"]
        condition_image = samples["condition_image"]
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        pipeline = VitonQformerPipeline(self)
        results = pipeline.generate(
            images=images,
            masks=masks,
            prompts=prompts,
            instructions=instructions,
            condition_images=condition_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            eta=eta
        )
        return results

    def prepare_inputs(self, samples, condition_image=None, target_image=None):
        condition_image = condition_image or self.condition_image
        target_image = target_image or self.target_image
        bz = len(samples["garm_instruction"])
        result = dict()
        if condition_image == "garm":
            result.update(
                {
                    "condition_image": samples["garm_vit_image"],
                    "instruction": samples["garm_instruction"]
                }
            )
        elif condition_image == "viton":
            result.update(
                {
                    "condition_image": samples["vton_vit_image"],
                    "instruction": samples["vton_instruction"]
                }
            )
        else:  # random
            condition_images = []
            instructions = []
            for i in range(bz):
                if random.random() < 0.5:
                    condition_images.append(samples["garm_vit_image"][i])
                    instructions.append(samples["garm_instruction"][i])
                else:
                    condition_images.append(samples["vton_vit_image"][i])
                    instructions.append(samples["vton_instruction"][i])
            result.update(
                {
                    "condition_image": torch.stack(condition_images).contiguous(),
                    "instruction": instructions
                }
            )
        if target_image == "garm":
            result.update(
                {
                    "image": samples["garm_image"],
                    "caption": samples["garm_caption"],
                    "mask": samples["garm_mask_image"]
                }
            )
        elif target_image == "viton":
            result.update(
                {
                    "image": samples["vton_image"],
                    "caption": samples["vton_caption"],
                    "mask": samples["vton_mask_image"]
                }
            )
        else:  # random
            images, captions, masks = [], [], []
            for i in range(bz):
                if random.random() < 0.5:
                    images.append(samples["garm_image"][i])
                    captions.append(samples["garm_caption"][i])
                    masks.append(samples["garm_mask_image"][i])
                else:
                    images.append(samples["vton_image"][i])
                    captions.append(samples["vton_caption"][i])
                    masks.append(samples["vton_mask_image"][i])
            result.update(
                {

                    "image": torch.stack(images).contiguous(),
                    "caption": captions,
                    "mask": torch.stack(masks).contiguous()
                }
            )

        return result

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):

        pretrained_model_name_or_path = cfg.get("pretrained_model_name_or_path",
                                                "stable-diffusion-v1-5/stable-diffusion-v1-5")
        tokenizer_name = cfg.get("tokenizer_name", None)
        revision = cfg.get("revision", None)
        variant = cfg.get("variant", None)
        gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        enable_xformers_memory_efficient_attention = cfg.get("enable_xformers_memory_efficient_attention", False)
        compute_dtype = cfg.get("compute_dtype", "fp16")
        proportion_empty_prompts = cfg.get("proportion_empty_prompts", 0)
        lora_config = cfg.get("lora_config", None)
        # q-former
        num_query_token = cfg.get("num_query_token", 32)
        max_txt_len = cfg.get("max_txt_len", 128)
        qformer_model_name_or_path = cfg.get("qformer_model_name_or_path", None)
        # q-former visual encoder
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_model_ckpt = cfg.get("vit_model_ckpt", None)
        img_size = cfg.get("img_size", 224)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_qformer = cfg.get("freeze_qformer", False)
        freeze_text_encoder = cfg.get("freeze_text_encoder", False)
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_vit_ln = cfg.get("freeze_vit_ln", False)
        condition_image = cfg.get("condition_image", "random")
        target_image = cfg.get("target_image", "random")

        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_name=tokenizer_name,
            revision=revision,
            variant=variant,
            gradient_checkpointing=gradient_checkpointing,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            compute_dtype=compute_dtype,
            proportion_empty_prompts=proportion_empty_prompts,
            lora_config=lora_config,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            qformer_model_name_or_path=qformer_model_name_or_path,
            vit_model=vit_model,
            vit_model_ckpt=vit_model_ckpt,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_qformer=freeze_qformer,
            freeze_text_encoder=freeze_text_encoder,
            freeze_vit=freeze_vit,
            freeze_vit_ln=freeze_vit_ln,
            condition_image=condition_image,
            target_image=target_image,
        )
        model.load_checkpoint_from_config(cfg)
        return model
