import torch
from transformers import VitsTokenizer
from typing import Optional

from vigc.common.registry import registry
from vigc.models.gan_base_model import GanBaseModel
from vigc.datasets.datasets.tts.vits_feature_extractor import VitsFeatureExtractor

from vigc.models.vits.losses import (
    kl_loss,
    generator_loss,
    discriminator_loss,
    feature_loss,
)

from .maximum_path import maximum_path
from .configuration_vits import VitsConfig
from .modeling_vits import VitsModelForPreTraining, VitsDiscriminator, slice_segments


@registry.register_model("vits")
class Vits(GanBaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/vits/vits.yaml",
    }

    def __init__(
            self,
            *,
            generator_model_name: str,
            discriminator_model_name: str,
            num_speakers: Optional[int] = None,
            override_vocabulary_embeddings: bool = False,
            gradient_checkpointing: bool = False,
            weight_disc=3.0,
            weight_duration=1.0,
            weight_kl=1.5,
            weight_mel=35.0,
            weight_gen=1.0,
            weight_fmaps=1.0
    ):
        super().__init__()
        self.config = VitsConfig.from_pretrained(generator_model_name)
        self.tokenizer = VitsTokenizer.from_pretrained(generator_model_name)
        self.feature_extractor = VitsFeatureExtractor.from_pretrained(generator_model_name)
        self.generator = VitsModelForPreTraining.from_pretrained(
            generator_model_name, config=self.config
        )
        del self.generator.discriminator
        self.generator.apply_weight_norm()

        if num_speakers is not None and self.config.num_speakers != num_speakers and num_speakers > 1:
            self.generator.resize_speaker_embeddings(
                num_speakers,
                self.config.speaker_embedding_size if self.config.speaker_embedding_size > 1 else 256
            )
        if override_vocabulary_embeddings:
            new_num_tokens = len(self.tokenizer)
            self.generator.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=2)

        if gradient_checkpointing:
            self.generator.gradient_checkpointing_enable()
        self.discriminator = VitsDiscriminator.from_pretrained(discriminator_model_name)
        self.discriminator.apply_weight_norm()

        self.weight_disc = weight_disc
        self.weight_duration = weight_duration
        self.weight_kl = weight_kl
        self.weight_mel = weight_mel
        self.weight_gen = weight_gen
        self.weight_fmaps = weight_fmaps

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        batch_size = len(samples["texts"])
        inputs = {
            "input_ids": samples["input_ids"],
            "attention_mask": samples["attention_mask"],
            "speaker_id": samples["speaker_id"],
        }
        with self.maybe_autocast():
            outputs = self.generator(**inputs)

        all_results = []
        for i in range(batch_size):
            audio = outputs.waveform[i, :outputs.sequence_lengths[i]].cpu().numpy()
            sampling_rate = self.config.sampling_rate
            audio = {
                "audio": audio,
                "sampling_rate": sampling_rate,
            }
            all_results.append(audio)

        return all_results

    def forward_common(self, samples, **kwargs):
        input_ids, attention_mask, labels, labels_attention_mask, speaker_id = samples["input_ids"], samples[
            "attention_mask"], samples["labels"], samples["labels_attention_mask"], samples["speaker_id"]
        with self.maybe_autocast():
            model_outputs = self.generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_attention_mask=labels_attention_mask,
                speaker_id=speaker_id,
                return_dict=True,
                monotonic_alignment_function=maximum_path,
            )
        mel_scaled_labels = samples["mel_scaled_input_features"]
        mel_scaled_target = slice_segments(mel_scaled_labels, model_outputs.ids_slice, self.generator.segment_size)
        mel_scaled_generation = self.feature_extractor._torch_extract_fbank_features(
            model_outputs.waveform.squeeze(1)
        )[1]
        target_waveform = samples["waveform"].transpose(1, 2)
        target_waveform = slice_segments(
            target_waveform, model_outputs.ids_slice * self.feature_extractor.hop_length, self.config.segment_size
        )

        return model_outputs, mel_scaled_generation, mel_scaled_target, target_waveform

    def forward_generator(self, *args, **kwargs):
        model_outputs, mel_scaled_generation, mel_scaled_target, target_waveform = args
        _, fmaps_target = self.discriminator(target_waveform)
        discriminator_candidate, fmaps_candidate = self.discriminator(model_outputs.waveform)

        loss_duration = torch.sum(model_outputs.log_duration)
        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
        loss_kl = kl_loss(
            model_outputs.prior_latents,
            model_outputs.posterior_log_variances,
            model_outputs.prior_means,
            model_outputs.prior_log_variances,
            model_outputs.labels_padding_mask,
        )
        loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
        loss_gen, losses_gen = generator_loss(discriminator_candidate)

        total_generator_loss = (
                loss_duration * self.weight_duration
                + loss_mel * self.weight_mel
                + loss_kl * self.weight_kl
                + loss_fmaps * self.weight_fmaps
                + loss_gen * self.weight_gen
        )
        return {"loss": total_generator_loss}

    def forward_discriminator(self, *args, **kwargs):
        model_outputs, mel_scaled_generation, mel_scaled_target, target_waveform = args
        discriminator_target, _ = self.discriminator(target_waveform)
        discriminator_candidate, _ = self.discriminator(model_outputs.waveform.detach())

        loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
            discriminator_target, discriminator_candidate
        )
        return {"loss": loss_disc * self.weight_disc}

    def forward(self, stage, *args, **kwargs):
        assert stage in ("common", "generator", "discriminator")
        if stage == "common":
            return self.forward_common(*args, **kwargs)
        elif stage == "generator":
            return self.forward_generator(*args, **kwargs)
        elif stage == "discriminator":
            return self.forward_discriminator(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        model_params = cfg.params

        model = cls(
            generator_model_name=model_params.generator_model_name,
            discriminator_model_name=model_params.discriminator_model_name,
            num_speakers=model_params.num_speakers,
            override_vocabulary_embeddings=model_params.override_vocabulary_embeddings,
            gradient_checkpointing=model_params.gradient_checkpointing,
            weight_disc=model_params.weight_disc,
            weight_duration=model_params.weight_duration,
            weight_kl=model_params.weight_kl,
            weight_mel=model_params.weight_mel,
            weight_gen=model_params.weight_gen,
            weight_fmaps=model_params.weight_fmaps,
        )
        model.load_checkpoint_from_config(cfg)
        return model
