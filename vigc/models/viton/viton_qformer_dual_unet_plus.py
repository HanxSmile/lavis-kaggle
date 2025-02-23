import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from typing import Literal
from transformers import AutoTokenizer
from vigc.models.viton.modules import CLIPTextModel, UNetVton2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from vigc.pipelines import VitonQformerDualUnetPipeline
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base
import contextlib
from vigc.models.viton.modules.attn_processors import VitonAttnProcessor
from vigc.models.viton.modules.attn_processors.utils import is_torch2_available

if is_torch2_available():
    from vigc.models.viton.modules.attn_processors import AttnProcessor2_0 as AttnProcessor
else:
    from vigc.models.viton.modules.attn_processors import AttnProcessor


@registry.register_model("viton_qformer_dual_unet_plus")
class VitonQformerDualUnetPlus(Blip2Base):
    """
    Viton Qformer Dual Unet Attention Plus model
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/viton/viton_qformer_dual_unet_plus.yaml",
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
            target_image: Literal['viton', 'garm', 'both'] = "both",
            cat_garm_mask: bool = False,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # assert (not freeze_qformer) and (not freeze_text_encoder)
        assert compute_dtype in ["fp16", "fp32", "bf16"]
        self.compute_dtype = torch.float32
        assert target_image in ["viton", "garm", "both"]
        self.target_image = target_image
        self.cat_garm_mask = cat_garm_mask
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

        self.vton_unet = UNetVton2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        self.vton_unet.replace_first_conv_layer(4 + 4)
        self.garm_unet = UNetVton2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        if self.cat_garm_mask:
            self.garm_unet.replace_first_conv_layer(4 + 1)

        self.vton_adapters = self.register_attn_processors(self.vton_unet)
        self.garm_adapters = self.register_attn_processors(self.garm_unet)

        self.text_encoder = self.freeze_module(self.text_encoder, "text_encoder", prevent_training_model=False).to(
            self.compute_dtype)

        self.vae = self.freeze_module(self.vae, "vae").to(self.compute_dtype)

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.inference_noise_scheduler = UniPCMultistepScheduler.from_config(self.noise_scheduler.config)

        if enable_xformers_memory_efficient_attention:
            self.vton_unet.enable_xformers_memory_efficient_attention()
            self.garm_unet.enable_xformers_memory_efficient_attention()
        if gradient_checkpointing:
            self.vton_unet.enable_gradient_checkpointing()
            self.garm_unet.enable_gradient_checkpointing()

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

    def register_attn_processors(self, unet):
        attn_procs = {}
        unet_sd = unet.state_dict()
        adapters = []
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
            if cross_attention_dim is not None:  # cross attention
                attn_procs[name] = AttnProcessor()
            else:  # self attention
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_q.weight": unet_sd[layer_name + ".to_q.weight"],
                }
                attn_procs[name] = VitonAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights, strict=False)
                adapters.append(attn_procs[name])
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(adapters)
        return adapter_modules

    @staticmethod
    def setup_mask(adapters, mask, condition_flag):
        for adapter in adapters:
            if isinstance(adapter, VitonAttnProcessor):
                adapter.setup_mask(mask)
                adapter.setup_condition_flag(condition_flag)

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

    def get_multi_modal_embeds(
            self, caption, instruction, image,
            training=True, negative_flag=False, negative_prompt=None):
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

    def forward_vton(self, samples):
        vton_samples_1 = self.prepare_inputs(samples, condition_image="garm", target_image="viton")
        garm_samples_0 = self.prepare_inputs(samples, condition_image="garm", target_image="garm")
        vton_encoder_hidden_states_1 = self.get_multi_modal_embeds(
            vton_samples_1["caption"],
            vton_samples_1["instruction"],
            vton_samples_1["condition_image"])
        garm_encoder_hidden_states_0 = self.get_multi_modal_embeds(
            garm_samples_0["caption"],
            garm_samples_0["instruction"],
            garm_samples_0["condition_image"])
        # Convert images to latent space
        with self.maybe_autocast(self.compute_dtype):
            vton_condition_latents = self.vae.encode(samples["agnostic_vton_image"]).latent_dist.mode()
            vton_latents = self.vae.encode(
                vton_samples_1["image"]).latent_dist.sample() * self.vae.config.scaling_factor
            garm_latents = self.vae.encode(
                garm_samples_0["image"]).latent_dist.sample() * self.vae.config.scaling_factor
            _, _, height, width = garm_latents.shape
        garm_mask = torch.nn.functional.interpolate(
            garm_samples_0["mask"], size=(height, width)
        )
        # Sample noise that we'll add to the latents
        vton_noise = torch.randn_like(vton_latents)
        bsz = vton_latents.shape[0]
        # Sample a random timestep for each image
        vton_timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                       device=self.device).long()
        zero_timesteps = torch.zeros_like(vton_timesteps).long()

        noisy_vton_latents = self.noise_scheduler.add_noise(vton_latents.float(), vton_noise.float(), vton_timesteps)
        no_noisy_garm_latents = self.noise_scheduler.add_noise(garm_latents.float(), vton_noise.float(), zero_timesteps)
        self.setup_mask(self.vton_adapters, vton_samples_1["mask"], condition_flag=False)
        self.setup_mask(self.garm_adapters, garm_samples_0["mask"], condition_flag=True)
        if self.cat_garm_mask:
            no_noisy_garm_latents = torch.cat([no_noisy_garm_latents, garm_mask], dim=1)
        with self.maybe_autocast(self.compute_dtype):
            _, garm_spatial_attn_outputs = self.garm_unet(
                no_noisy_garm_latents,
                [],
                0,
                encoder_hidden_states=garm_encoder_hidden_states_0,
                return_dict=False,
            )
            vton_model_pred, _ = self.vton_unet(
                torch.cat([noisy_vton_latents, vton_condition_latents], dim=1),
                garm_spatial_attn_outputs,
                vton_timesteps,
                encoder_hidden_states=vton_encoder_hidden_states_1,
                return_dict=False,
            )
            vton_model_pred = vton_model_pred[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            vton_target = vton_noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            vton_target = self.noise_scheduler.get_velocity(vton_latents, vton_noise, vton_timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        vton_loss = F.mse_loss(vton_model_pred.float(), vton_target.float(), reduction="mean")
        return vton_loss

    def forward_garm(self, samples):
        vton_samples_0 = self.prepare_inputs(samples, condition_image="viton", target_image="viton")
        garm_samples_1 = self.prepare_inputs(samples, condition_image="viton", target_image="garm")
        vton_encoder_hidden_states_0 = self.get_multi_modal_embeds(vton_samples_0["caption"],
                                                                   vton_samples_0["instruction"],
                                                                   vton_samples_0["condition_image"])
        garm_encoder_hidden_states_1 = self.get_multi_modal_embeds(garm_samples_1["caption"],
                                                                   garm_samples_1["instruction"],
                                                                   garm_samples_1["condition_image"])
        # Convert images to latent space
        with self.maybe_autocast(self.compute_dtype):
            vton_condition_latents = self.vae.encode(samples["agnostic_vton_image"]).latent_dist.mode()
            vton_latents = self.vae.encode(
                vton_samples_0["image"]).latent_dist.sample() * self.vae.config.scaling_factor
            garm_latents = self.vae.encode(
                garm_samples_1["image"]).latent_dist.sample() * self.vae.config.scaling_factor
            _, _, height, width = garm_latents.shape

        garm_mask = torch.nn.functional.interpolate(
            garm_samples_1["mask"], size=(height, width)
        )
        garm_noise = torch.randn_like(garm_latents)
        bsz = vton_latents.shape[0]
        # Sample a random timestep for each image
        garm_timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                       device=self.device).long()
        zero_timesteps = torch.zeros_like(garm_timesteps).long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        no_noisy_vton_latents = self.noise_scheduler.add_noise(vton_latents.float(), garm_noise.float(), zero_timesteps)
        noisy_garm_latents = self.noise_scheduler.add_noise(garm_latents.float(), garm_noise.float(), garm_timesteps)
        self.setup_mask(self.vton_adapters, vton_samples_0["mask"], condition_flag=True)
        self.setup_mask(self.garm_adapters, garm_samples_1["mask"], condition_flag=False)
        if self.cat_garm_mask:
            noisy_garm_latents = torch.cat([noisy_garm_latents, garm_mask], dim=1)
        with self.maybe_autocast(self.compute_dtype):
            _, vton_spatial_attn_outputs = self.vton_unet(
                torch.cat([no_noisy_vton_latents, vton_condition_latents], dim=1),
                [],
                0,
                encoder_hidden_states=vton_encoder_hidden_states_0,
                return_dict=False,
            )
            garm_model_pred, _ = self.garm_unet(
                noisy_garm_latents,
                vton_spatial_attn_outputs,
                garm_timesteps,
                encoder_hidden_states=garm_encoder_hidden_states_1,
                return_dict=False,
            )
            garm_model_pred = garm_model_pred[0]
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            garm_target = garm_noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            garm_target = self.noise_scheduler.get_velocity(garm_latents, garm_noise, garm_timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        garm_loss = F.mse_loss(garm_model_pred.float(), garm_target.float(), reduction="mean")
        return garm_loss

    def forward(self, samples):
        if self.target_image == "viton":
            loss = self.forward_viton(samples)
        elif self.target_image == "garm":
            loss = self.forward_garm(samples)
        else:
            if random.random() < 0.5:
                loss = self.forward_garm(samples)
            else:
                loss = self.forward_vton(samples)
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
            if isinstance(finetune_path, str):
                finetune_path = [finetune_path]
            for p in finetune_path:
                self.load_checkpoint(url_or_filename=p)
                logging.info(f"Loaded finetuned model '{p}'.")

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
            vae_encode_method: str = "mode"
    ):
        agnostic_vton_images = samples["agnostic_vton_image"]
        target_samples = self.prepare_inputs(samples, condition_image=condition_image, target_image=target_image)
        condition_samples = self.prepare_inputs(samples, condition_image=condition_image, target_image=condition_image)

        generator = None if seed is None else torch.Generator().manual_seed(seed)
        pipeline = VitonQformerDualUnetPipeline(self)
        results = pipeline.generate(
            agnostic_vton_images=agnostic_vton_images,
            target_inputs=target_samples,
            condition_inputs=condition_samples,
            target_type=target_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            eta=eta,
            vae_encode_method=vae_encode_method
        )
        return results

    def prepare_inputs(self, samples, condition_image=None, target_image=None):
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
        target_image = cfg.get("target_image", "both")
        cat_garm_mask = cfg.get("cat_garm_mask", False)

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
            target_image=target_image,
            cat_garm_mask=cat_garm_mask,
        )
        model.load_checkpoint_from_config(cfg)
        return model
