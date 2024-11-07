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


@registry.register_processor("rag_caption")
class RAGCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=256):
        super().__init__()
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

    def pre_caption(self, x):
        x = x.lower()  # Convert words to lowercase
        x = re.sub("@\w+", '', x)  # Delete strings starting with @
        # x = re.sub("'\d+", '',x)      # Delete Numbers
        x = re.sub("http\w+", '', x)  # Delete URL
        x = re.sub(r"\\\(", " ", x)
        x = re.sub(r"\\\)", " ", x)
        x = re.sub(r"[ ]{1,}", " ", x)
        x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\,+", ",", x)
        x = x.strip()  # Remove empty characters at the beginning and end
        return x
