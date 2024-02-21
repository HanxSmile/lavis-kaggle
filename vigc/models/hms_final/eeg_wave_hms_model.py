import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
import torch.nn as nn
import torch.nn.functional as F


class ResNet_1D_Block(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout,
            downsampling
    ):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


@registry.register_model("eeg_wave_hms")
class EEGWaveHMSClassifier(BaseModel):
    """
    copied from https://www.kaggle.com/code/medali1992/hms-resnet1d-gru-train
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/eeg_wave_hms_classifier.yaml",
    }

    def __init__(
            self,
            kernels,
            in_channels=20,
            fixed_kernel_size=17,
            dropout=0.
    ):
        super().__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                                 stride=1, padding=0, bias=False, )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            padding=fixed_kernel_size // 2,
            dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features=424, out_features=6)
        )

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def _make_resnet_layer(self, kernel_size, stride, blocks=9, padding=0, dropout=0.):
        layers = []

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(ResNet_1D_Block(
                in_channels=self.planes,
                out_channels=self.planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                downsampling=downsampling))

        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

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
            **kwargs
    ):
        model_inputs, label, eeg_id = samples["eeg"], samples["label"], samples["eeg_id"]
        with self.maybe_autocast():
            features = self.extract_features(model_inputs)
            logits = self.head(features)
            probs = F.softmax(logits, dim=1)  # [b, 6]
            # loss = self.eval_criterion(logits, label).sum(dim=1)  # [b, 1]

        return {"result": probs, "label": label, "eeg_id": eeg_id}

    def forward(self, samples, **kwargs):
        model_inputs, label = samples["eeg"], samples["label"]
        with self.maybe_autocast():
            features = self.extract_features(model_inputs)
            logits = self.head(features)
            log_prob_logits = F.log_softmax(logits, dim=1)
            loss = self.kl_loss(log_prob_logits, label)
        return {"loss": loss}

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

        kernels = cfg.get("kernels")
        in_channels = cfg.get("in_channels", 20)
        fixed_kernel_size = cfg.get("fixed_kernel_size", 17)
        dropout = cfg.get("dropout", 0.)

        model = cls(
            kernels=kernels,
            in_channels=in_channels,
            fixed_kernel_size=fixed_kernel_size,
            dropout=dropout
        )
        model.load_checkpoint_from_config(cfg)
        return model
