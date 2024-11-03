import logging
import torch
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import segmentation_models_pytorch as smp
import contextlib
import torchvision.transforms.functional as F


@registry.register_model("unet_semantic_segmentation")
class UnetSemanticSegmentationModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/semantic_segmentation/unet_semantic_segmentation.yaml",
    }

    def __init__(
            self,
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=-1,
            activation=None
    ):
        super().__init__()
        assert classes > 0
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
            in_channels=in_channels,
        )

        self.JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
        self.DiceLoss = smp.losses.DiceLoss(mode='multilabel')
        self.BCELoss = smp.losses.SoftBCEWithLogitsLoss(ignore_index=-100)
        self.LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
        self.TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False, ignore_index=-100)

    def criterion(self, y_pred, y_true):
        return 0.5 * self.BCELoss(y_pred, y_true) + 0.5 * self.TverskyLoss(y_pred, y_true)

    def dice_coef(self, y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2 * inter + epsilon) / (den + epsilon))
        return dice

    def iou_coef(self, y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
        iou = ((inter + epsilon) / (union + epsilon))
        return iou

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
            image_size=None,
            threshold=0.5,
            **kwargs
    ):
        if isinstance(image_size, int):
            image_size = [image_size, image_size]  # H, W

        images, masks = samples["image"], samples["mask"]

        with self.maybe_autocast():
            y_pred = self.model(images)

        y_pred = torch.sigmoid(y_pred)

        y_h, y_w = images.shape[-2:]
        m_h, m_w = masks.shape[-2:]
        if image_size is not None:
            if y_h != image_size[0] or y_w != image_size[1]:
                y_pred = F.resize(y_pred, list(image_size))
            if m_h != image_size[0] or m_w != image_size[1]:
                masks = F.resize(masks, list(image_size), F.InterpolationMode.NEAREST)
        else:
            if y_h != m_h or y_w != m_w:
                y_pred = F.resize(y_pred, [m_h, m_w])

        val_dice = self.dice_coef(masks, y_pred, thr=threshold).cpu().detach().numpy()
        val_jaccard = self.iou_coef(masks, y_pred, thr=threshold).cpu().detach().numpy()
        return {
            "dice_score": val_dice,
            "iou_score": val_jaccard,
            "y_pred": y_pred,
        }

    def forward(self, samples, **kwargs):
        images, masks = samples["image"], samples["mask"]
        with self.maybe_autocast():
            y_pred = self.model(images)

        loss = self.criterion(y_pred, masks)
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
        encoder_name = cfg.get("encoder_name", "efficientnet-b1")
        encoder_weights = cfg.get("encoder_weights", "imagenet")
        in_channels = cfg.get("in_channels", 3)
        classes = cfg.get("classes", -1)
        activation = cfg.get("activation", None)
        model = cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        model.load_checkpoint_from_config(cfg)
        return model
