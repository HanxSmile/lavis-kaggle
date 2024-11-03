"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.common.registry import registry
from vigc.processors.base_processor import BaseProcessor
import albumentations as A
import cv2
from .semantic_seg_utils import Pad, Resize, RandomCrop


@registry.register_processor("semantic_segmentation_train")
class SemanticSegmentationTrainProcessor(BaseProcessor):
    def __init__(
            self, image_size=384, crop_size=384, ratio_range=(0.75, 1.5)
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.resize_transform = Resize(image_size, tuple(ratio_range))
        self.crop_transform = RandomCrop(crop_size)
        self.pad_transform = Pad(crop_size)

        self.transform = A.Compose([
            A.Resize(*crop_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=crop_size[0] // 20, max_width=crop_size[1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0)

    def __call__(self, **kwargs):
        data = self.resize_transform(**kwargs)
        data = self.crop_transform(**data)
        data = self.pad_transform(**data)
        return self.transform(**data)

    @classmethod
    def from_config(cls, cfg=None):
        image_size = cfg.get("image_size", [384, 384])
        crop_size = cfg.get("crop_size", [384, 384])
        ratio_range = cfg.get("ratio_range", [0.75, 1.25])
        return cls(
            image_size=image_size,
            crop_size=crop_size,
            ratio_range=ratio_range,
        )


@registry.register_processor("semantic_segmentation_eval")
class SemanticSegmentationEvalProcessor(BaseProcessor):
    def __init__(self, image_size=384, mask_size=384):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(mask_size, int):
            mask_size = (mask_size, mask_size)
        self.image_size = image_size
        self.mask_size = mask_size
        self.image_transform = A.Compose([
            A.Resize(*image_size, interpolation=cv2.INTER_LINEAR),
        ], p=1.0)

        self.mask_transform = A.Compose([
            A.Resize(*mask_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)

    def __call__(self, **kwargs):
        image, mask = kwargs['image'], kwargs['mask']
        img_h, img_w = image.shape[:2]
        msk_h, msk_w = mask.shape[:2]
        if img_h != self.image_size[0] or img_w != self.image_size[1]:
            image = self.image_transform(**kwargs)["image"]
        if msk_h != self.mask_size[0] or msk_w != self.mask_size[1]:
            mask = self.mask_transform(**kwargs)["mask"]
        return {"image": image, "mask": mask}

    @classmethod
    def from_config(cls, cfg=None):
        image_size = cfg.get("image_size", [384, 384])
        mask_size = cfg.get("mask_size", [384, 384])

        return cls(
            image_size=image_size,
            mask_size=mask_size,
        )
