from torch.utils.data import Dataset as torch_Dataset
import pandas as pd
import os
import re
import os.path as osp
import datasets
import librosa
import torch
from typing import Dict, List, Union
from bnunicodenormalizer import Normalizer
from datasets import Audio
import numpy as np

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


class BengaliCVBN(torch_Dataset):
    DATASET_NAME = "/mnt/petrelfs/share_data/hanxiao/cvbn"

    def __init__(self, processor, split: str, transform=None):
        split = split.lower()
        assert split in ("train", "validation")
        self.inner_dataset = datasets.load_from_disk(self.DATASET_NAME)[split]
        self.inner_dataset = self.inner_dataset.remove_columns(
            ['up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'])
        self.inner_dataset = self.inner_dataset.cast_column("audio", Audio(sampling_rate=16_000))

        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        audio = ann["audio"]
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        input_features = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_features[0]

        sentence = normalize(remove_special_characters(ann["sentence"]))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_features": input_features, "labels": labels, "sentence": sentence, "id": str(index)}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        result = {}
        result["input_features"] = batch["input_features"]
        result["labels"] = labels
        result["sentences"] = [_["sentence"] for _ in features]
        result["ids"] = [_["id"] for _ in features]
        return result


class BengaliASR(torch_Dataset):
    def __init__(self, feature_extractor, tokenizer, processor, data_root, split: str, max_label_length: int,
                 transform=None):
        split = split.lower()
        assert split in ("train", "valid")
        self.processor = processor
        self.audio_processor = feature_extractor
        self.text_processor = tokenizer
        self.media_root = osp.join(data_root, "train_mp3s")
        self.anno_path = osp.join(data_root, "train.csv")
        annotations = pd.read_csv(self.anno_path)
        data = annotations[annotations["split"] == split]
        data["audio"] = self.media_root + os.sep + data["id"] + ".mp3"
        self.inner_dataset = data
        self.transform = transform
        self.max_label_length = max_label_length

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
        input_features = self.audio_processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.text_processor(sentence, truncation=True, max_length=self.max_label_length).input_ids

        return {"input_features": input_features, "labels": labels, "sentence": sentence, "id": ann.id}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        result = {}
        result["input_features"] = batch["input_features"]
        result["labels"] = labels
        result["sentences"] = [_["sentence"] for _ in features]
        result["ids"] = [_["id"] for _ in features]
        return result


class BengaliASRTest(torch_Dataset):
    def __init__(self, feature_extractor, tokenizer, processor, data_root, max_label_length: int):
        self.processor = processor
        self.audio_processor = feature_extractor
        self.text_processor = tokenizer
        self.media_root = osp.join(data_root, "examples")
        self.anno_path = osp.join(data_root, "annoated.csv")
        self.inner_dataset = pd.read_csv(self.anno_path, sep="\t")
        self.max_label_length = max_label_length

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
        input_features = self.audio_processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.text_processor(sentence, truncation=True, max_length=self.max_label_length).input_ids

        return {"input_features": input_features, "labels": labels, "sentence": ann.sentence, "id": ann.file,
                "audio": audio}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        result = {}
        result["input_features"] = batch["input_features"]
        result["labels"] = labels
        result["sentences"] = [_["sentence"] for _ in features]
        result["ids"] = [_["id"] for _ in features]
        result["raw_audios"] = [_["audio"] for _ in features]
        return result
