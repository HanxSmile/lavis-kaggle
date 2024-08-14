import torch
from vigc.common.registry import registry
from vigc.models.gan_base_model import GanBaseModel
import torch.nn.functional as F
from vigc.models.gpt_sovits.utils import (
    spec_to_mel_torch,
    slice_segments,
    mel_spectrogram_torch,
)
from vigc.models.gpt_sovits.losses import (
    kl_loss,
    generator_loss,
    discriminator_loss,
    feature_loss,
)
from vigc.models.gpt_sovits.vits import SynthesizerTrn, MultiPeriodDiscriminator


@registry.register_model("vq_vits")
class VQVits(GanBaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/gpt-vits/vits.yaml",
    }

    def __init__(
            self,
            *,
            # generator
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            symbols_nums,
            n_speakers=0,
            gin_channels=0,
            use_sdp=True,
            semantic_frame_rate=None,
            freeze_quantizer=None,
            # discriminator
            use_spectral_norm,
            # learning_rate
            text_low_lr_rate,
            # mel hps
            mel_hps,
    ):
        super().__init__()
        self.generator = SynthesizerTrn(
            spec_channels=spec_channels,
            segment_size=segment_size,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            symbols_nums=symbols_nums,
            n_speakers=n_speakers,
            gin_channels=gin_channels,
            use_sdp=use_sdp,
            semantic_frame_rate=semantic_frame_rate,
            freeze_quantizer=freeze_quantizer,
        )
        self.discriminator = MultiPeriodDiscriminator(
            use_spectral_norm=use_spectral_norm
        )
        self.text_low_lr_rate = text_low_lr_rate
        self.mel_hps = mel_hps

    def get_generator_parameter_group(self, learning_rate):
        te_p = list(map(id, self.generator.enc_p.text_embedding.parameters()))
        et_p = list(map(id, self.generator.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, self.generator.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            self.generator.parameters(),
        )
        params = [
            {
                "params": base_params, "lr": learning_rate},
            {
                "params": self.generator.enc_p.text_embedding.parameters(),
                "lr": learning_rate * self.text_low_lr_rate,
            },
            {
                "params": self.generator.enc_p.encoder_text.parameters(),
                "lr": learning_rate * self.text_low_lr_rate,
            },
            {
                "params": self.generator.enc_p.mrte.parameters(),
                "lr": learning_rate * self.text_low_lr_rate,
            },
        ]
        return params

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = samples["ssl"], samples["ssl_lengths"], \
            samples["spec"], samples["spec_lengths"], samples["y"], samples["y_lengths"], samples["text"], samples[
            "text_lengths"]
        with self.maybe_autocast():
            y_hat, mask, *_ = self.generator.infer(ssl, spec, spec_lengths, text, text_lengths)
        y_hat_lengths = mask.sum([1, 2]).long() * self.mel_hps.hop_length
        batch_size = y_hat.shape[0]
        all_results = []
        for i in range(batch_size):
            audio = y_hat[i, :, :y_hat_lengths[i]]
            all_results.append(audio)

        return all_results

    def forward_common(self, samples, **kwargs):
        ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = samples["ssl"], samples["ssl_lengths"], \
            samples["spec"], samples["spec_lengths"], samples["y"], samples["y_lengths"], samples["text"], samples[
            "text_lengths"]
        with self.maybe_autocast():
            (
                y_hat,
                kl_ssl,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                stats_ssl,
            ) = self.generator(ssl, spec, spec_lengths, text, text_lengths)

            mel = spec_to_mel_torch(
                spec,
                self.mel_hps.filter_length,
                self.mel_hps.n_mel_channels,
                self.mel_hps.sampling_rate,
                self.mel_hps.mel_fmin,
                self.mel_hps.mel_fmax,
            )
            y_mel = slice_segments(
                mel, ids_slice, self.generator.segment_size // self.mel_hps.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.mel_hps.filter_length,
                self.mel_hps.n_mel_channels,
                self.mel_hps.sampling_rate,
                self.mel_hps.hop_length,
                self.mel_hps.win_length,
                self.mel_hps.mel_fmin,
                self.mel_hps.mel_fmax,
            )

            y = slice_segments(
                y, ids_slice * self.mel_hps.hop_length, self.generator.segment_size
            )  # slice

        return y, y_hat, y_mel, y_hat_mel, z_p, logs_q, m_p, logs_p, z_mask, kl_ssl

    def forward_generator(self, *args, **kwargs):
        y, y_hat, y_mel, y_hat_mel, z_p, logs_q, m_p, logs_p, z_mask, kl_ssl = args
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.mel_hps.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.mel_hps.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
        return {"loss": loss_gen_all}

    def forward_discriminator(self, *args, **kwargs):
        y, y_hat, y_mel, y_hat_mel, *_ = args
        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc
        return {"loss": loss_disc_all}

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
            # generator
            spec_channels=model_params.spec_channels,
            segment_size=model_params.segment_size,
            inter_channels=model_params.inter_channels,
            hidden_channels=model_params.hidden_channels,
            filter_channels=model_params.filter_channels,
            n_heads=model_params.n_heads,
            n_layers=model_params.n_layers,
            kernel_size=model_params.kernel_size,
            p_dropout=model_params.p_dropout,
            resblock=model_params.resblock,
            resblock_kernel_sizes=model_params.resblock_kernel_sizes,
            resblock_dilation_sizes=model_params.resblock_dilation_sizes,
            upsample_rates=model_params.upsample_rates,
            upsample_initial_channel=model_params.upsample_initial_channel,
            upsample_kernel_sizes=model_params.upsample_kernel_sizes,
            symbols_nums=model_params.symbols_nums,
            n_speakers=model_params.get("n_speakers", 0),
            gin_channels=model_params.get("gin_channels", 0),
            use_sdp=model_params.get("use_sdp", True),
            semantic_frame_rate=model_params.get("semantic_frame_rate", None),
            freeze_quantizer=model_params.get("freeze_quantizer", None),
            # discriminator
            use_spectral_norm=model_params.use_spectral_norm,
            # learning_rate
            text_low_lr_rate=model_params.text_low_lr_rate,
            # mel hps
            mel_hps=model_params.mel_hps,
        )
        model.load_checkpoint_from_config(cfg)
        return model
