import logging
import torch
from vigc.models.blip2_models.blip2 import disabled_train
import timm
import torch.nn as nn
from vigc.models.hms_classifier.modules import GeM


class ImageHMSFeatureExtractor(nn.Module):
    """
    copied from https://www.kaggle.com/code/crackle/efficientnetb0-pytorch-starter-lb-0-40/notebook
    """

    def __init__(
            self,
            model_name="tf_efficientnet_b0",
            use_kaggle_spectrograms=True,
            use_eeg_spectrograms=False,
            freeze_encoder=False,
            use_gem=False,
            dropout=0.,
            embedding_dim=256,
            tile_input=True,
    ):
        super().__init__()
        self.use_kaggle_spectrograms = use_kaggle_spectrograms
        self.use_eeg_spectrograms = use_eeg_spectrograms
        self.tile_input = tile_input
        in_channels = 3 if tile_input else 4
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=in_channels)
        num_in_features = self.backbone.get_classifier().in_features
        if freeze_encoder:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
            self.backbone = self.backbone.eval()
            self.backbone.train = disabled_train
            logging.warning("Freeze the backbone model")

        if use_gem:
            self.head = nn.Sequential(
                GeM(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(num_in_features, embedding_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(num_in_features, embedding_dim)
            )

    def preprocess_inputs(self, x):
        # x.shape = [b, 128, 256, 8]
        if self.tile_input:
            x1 = [x[:, :, :, i:i + 1] for i in range(4)]
            x1 = torch.cat(x1, dim=1)
            x2 = [x[:, :, :, i + 4:i + 5] for i in range(4)]
            x2 = torch.cat(x2, dim=1)

            if self.use_kaggle_spectrograms & self.use_eeg_spectrograms:
                x = torch.cat([x1, x2], dim=2)
            elif self.use_eeg_spectrograms:
                x = x2
            else:
                x = x1
            x = torch.cat([x, x, x], dim=3)
        else:
            x1 = x[..., :4]  # [b, 128, 256, 4]
            x2 = x[..., 4:]
            x = torch.cat([x1, x2], dim=1)  # [b, 256, 256, 4]
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, samples, **kwargs):
        model_inputs = self.preprocess_inputs(samples["image"])

        features = self.backbone.forward_features(model_inputs)
        embedding = self.head(features)
        return embedding
