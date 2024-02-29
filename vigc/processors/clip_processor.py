"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from vigc.common.registry import registry
from vigc.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
import albumentations as A
from transformers import AutoProcessor
import numpy as np
from PIL import Image


def prepare_transform(image_size):
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        # A.OneOf([
        #     A.MotionBlur(blur_limit=5),
        #     A.MedianBlur(blur_limit=5),
        #     A.GaussianBlur(blur_limit=5),
        #     A.GaussNoise(var_limit=(5.0, 30.0)),
        # ], p=0.7),

        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=1.0),
        #     A.GridDistortion(num_steps=5, distort_limit=1.),
        #     A.ElasticTransform(alpha=3),
        # ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.2), max_w_size=int(image_size * 0.2), num_holes=1, p=0.7),
    ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
    ])
    return transforms_train, transforms_val


@registry.register_processor("clip_caption")
class ClipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=256):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 256)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # # truncate caption
        # caption_words = caption.split(" ")
        # if len(caption_words) > self.max_words:
        #     caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("clip_image_train")
class ClipImageTrainProcessor(BaseProcessor):
    def __init__(
            self, image_size=384, processor_name="openai/clip-vit-base-patch32"
    ):
        super().__init__()

        self.transform = prepare_transform(image_size)[0]
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def __call__(self, item):
        new_image = Image.fromarray(self.transform(image=np.array(item))["image"])
        return self.processor(images=new_image, return_tensors="pt")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        processor_name = cfg.get("processor_name", "openai/clip-vit-base-patch32")

        return cls(
            image_size=image_size,
            processor_name=processor_name
        )


@registry.register_processor("clip_image_eval")
class ClipImageEvalProcessor(BaseProcessor):
    def __init__(self, image_size=384, processor_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.transform = prepare_transform(image_size)[1]
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def __call__(self, item):
        new_image = Image.fromarray(self.transform(image=np.array(item))["image"])
        return self.processor(images=new_image, return_tensors="pt")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        processor_name = cfg.get("processor_name", "openai/clip-vit-base-patch32")

        return cls(
            image_size=image_size,
            processor_name=processor_name
        )
