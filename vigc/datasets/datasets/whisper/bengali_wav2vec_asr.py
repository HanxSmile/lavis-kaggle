from torch.utils.data import Dataset as torch_Dataset
import pandas as pd
import os
import os.path as osp
import numpy as np
import librosa
import torch
import re
from typing import Dict, List, Union
from bnunicodenormalizer import Normalizer

bnorm = Normalizer()

pd.options.mode.chained_assignment = None

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…]'


def remove_special_characters(sentence):
    sentence = re.sub(chars_to_ignore_regex, '', sentence) + " "
    return sentence


def normalize(sentence):
    _words = [bnorm(word)['normalized'] for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    return sentence


class Wav2VecBengaliASR(torch_Dataset):
    def __init__(self, processor, data_root, split: str, transform=None):
        split = split.lower()
        assert split in ("train", "valid")
        self.processor = processor
        self.media_root = osp.join(data_root, "train_mp3s")
        self.anno_path = osp.join(data_root, "train.csv")
        annotations = pd.read_csv(self.anno_path)
        data = annotations[annotations["split"] == split]
        data["audio"] = self.media_root + os.sep + data["id"] + ".mp3"
        self.inner_dataset = data
        self.transform = transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = ann.audio
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        array = np.trim_zeros(array, "fb")
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        input_values = self.processor(audio["array"], sampling_rate=16_000).input_values[0]
        input_length = len(input_values)
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann.id,
                "input_length": input_length}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["sentences"] = [_["sentence"] for _ in features]
        batch["ids"] = [_["id"] for _ in features]
        return batch


class Wav2VecBengaliASRTest(torch_Dataset):
    def __init__(self, processor, data_root):
        self.processor = processor
        self.media_root = osp.join(data_root, "examples")
        self.anno_path = osp.join(data_root, "annoated.csv")
        self.inner_dataset = pd.read_csv(self.anno_path, sep="\t")

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = osp.join(self.media_root, ann.file)
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        array = np.trim_zeros(array, "fb")
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        input_values = self.processor(audio["array"], sampling_rate=16_000).input_values[0]
        input_length = len(input_values)
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann.file,
                "input_length": input_length}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["sentences"] = [_["sentence"] for _ in features]
        batch["ids"] = [_["id"] for _ in features]
        return batch
