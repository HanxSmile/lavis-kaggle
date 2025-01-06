"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.common.registry import registry
from vigc.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from .text_rec_utils.rec_aug import RecAug
from .text_rec_utils.resize import RecResizeAndNormImg


@registry.register_processor("text_rec_train")
class TextRecTrainProcessor(BaseProcessor):
    def __init__(
            self,
            image_shape,
            tia_prob=0.4,
            crop_prob=0.4,
            reverse_prob=0.4,
            noise_prob=0.4,
            jitter_prob=0.4,
            blur_prob=0.4,
            hsv_aug_prob=0.4,
    ):
        super().__init__()
        self.aug_transform = RecAug(
            tia_prob=tia_prob,
            crop_prob=crop_prob,
            reverse_prob=reverse_prob,
            noise_prob=noise_prob,
            jitter_prob=jitter_prob,
            blur_prob=blur_prob,
            hsv_aug_prob=hsv_aug_prob,
        )
        self.resize_transform = RecResizeAndNormImg(
            image_shape=image_shape,
            eval_mode=False,
            padding=True
        )

    def __call__(self, img):
        img = self.aug_transform(img)
        img = self.resize_transform(img)
        return img

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_shape = cfg.image_shape
        tia_prob = cfg.get('tia_prob', 0.4)
        crop_prob = cfg.get('crop_prob', 0.4)
        reverse_prob = cfg.get('reverse_prob', 0.4)
        noise_prob = cfg.get('noise_prob', 0.4)
        jitter_prob = cfg.get('jitter_prob', 0.4)
        blur_prob = cfg.get('blur_prob', 0.4)
        hsv_aug_prob = cfg.get('hsv_aug_prob', 0.4)

        return cls(
            image_shape=image_shape,
            tia_prob=tia_prob,
            crop_prob=crop_prob,
            reverse_prob=reverse_prob,
            noise_prob=noise_prob,
            jitter_prob=jitter_prob,
            blur_prob=blur_prob,
            hsv_aug_prob=hsv_aug_prob,
        )


@registry.register_processor("text_rec_eval")
class TextRecEvalProcessor(BaseProcessor):
    def __init__(self, image_shape, batch_mode=False):
        super().__init__()

        self.resize_transform = RecResizeAndNormImg(
            image_shape=image_shape,
            eval_mode=True,
            batch_mode=batch_mode,
        )

    def __call__(self, img):
        return self.resize_transform(img)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(
            cfg.image_shape,
            batch_mode=cfg.get('batch_mode', False),
        )
