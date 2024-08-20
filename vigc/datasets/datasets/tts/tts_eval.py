from torch.utils.data import Dataset as torch_Dataset
import torch
from typing import Dict, List, Union
import os
from .romanize import uromanize
import json


class VitsTTSEval(torch_Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            speaker_nums=None,
            max_tokens_length=500,
            do_lower_case=False,
            uroman_path=None,
            language=None
    ):

        if speaker_nums is not None and speaker_nums == 1:
            speaker_nums = None
        assert speaker_nums is None or isinstance(speaker_nums, int)
        self.speaker_nums = speaker_nums
        with open(data_root, "r") as f:
            self.inner_dataset = json.load(f)
        self.tokenizer = tokenizer
        self.language = language
        self.max_tokens_length = max_tokens_length
        self.do_lower_case = do_lower_case
        self.is_uroman = tokenizer.is_uroman
        self.uroman_path = uroman_path
        if self.is_uroman:
            assert uroman_path is not None
            assert os.path.exists(uroman_path)

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        sample = self.inner_dataset[index]
        text = sample["text"].strip()
        input_str = text.lower() if self.do_lower_case else text
        if self.is_uroman:
            input_str = uromanize(input_str, self.uroman_path)
        string_inputs = self.tokenizer(input_str, return_attention_mask=False)
        if len(string_inputs) > self.max_tokens_length:
            return self[(index + 1) % len(self.inner_dataset)]
        input_ids = string_inputs[: self.max_tokens_length]
        tokens_input_length = len(input_ids)

        res = {
            "input_ids": input_ids,
            "tokens_input_length": tokens_input_length,
            "id": index,
            "text": text,
        }

        if self.speaker_nums is not None:
            res["speaker_nums"] = 0

        return res

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = "input_ids"
        input_ids = [{model_input_name: feature[model_input_name]} for feature in features]

        # pad input tokens
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", return_attention_mask=True)

        batch["speaker_id"] = (
            torch.tensor([feature["speaker_id"] for feature in features]) if "speaker_id" in features[0] else None
        )
        batch["ids"] = [_["id"] for _ in features]
        batch["texts"] = [_["text"] for _ in features]

        return batch
