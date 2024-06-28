from torch.utils.data import Dataset as torch_Dataset
import datasets
import torch
from typing import Dict, List, Union
from datasets import Audio, concatenate_datasets
import numpy as np

MIN_SECS = 1
MAX_SECS = 30
TARGET_SR = 16_000


class FleursTrain(torch_Dataset):
    def __init__(self, data_root, processor, transform=None, max_label_length=448, split="train"):
        if isinstance(split, str):
            split = [split]
        inner_dataset = datasets.load_from_disk(data_root)
        inner_dataset = concatenate_datasets([inner_dataset[_] for _ in split])
        self.inner_dataset = inner_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        self.processor = processor
        self.transform = transform
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.inner_dataset)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        return ann["audio"], ann["transcription"], str(index)

    def is_valid(self, input_values):
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return MAX_SECS > input_secs > MIN_SECS

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        return audio

    def __getitem__(self, index):
        audio, sentence, ann_id = self._parse_ann_info(index)
        audio = self.transform_array(audio)

        if not self.is_valid(audio["array"]):
            return self[(index + 1) % len(self)]  # filter too long or too short audio

        input_features = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        labels = self.processor.tokenizer(sentence, truncation=True, max_length=self.max_label_length).input_ids

        return {"input_features": input_features, "labels": labels, "sentence": sentence, "id": ann_id}

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


class FluersTest(torch_Dataset):
    def __init__(self, data_root, processor, max_label_length=448, split="test"):
        inner_dataset = datasets.load_from_disk(data_root)[split]
        self.inner_dataset = inner_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        self.processor = processor
        self.transform = None
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.inner_dataset)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        return ann["audio"], ann["transcription"], str(index)

    def is_valid(self, input_values):
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return MAX_SECS > input_secs > MIN_SECS

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        return audio

    def __getitem__(self, index):
        audio, sentence, ann_id = self._parse_ann_info(index)
        audio = self.transform_array(audio)
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])

        if not self.is_valid(audio["array"]):
            return self[(index + 1) % len(self)]  # filter too long or too short audio

        input_features = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        labels = self.processor.tokenizer(sentence, truncation=True, max_length=self.max_label_length).input_ids

        return {"input_features": input_features, "labels": labels, "sentence": sentence, "id": ann_id, "audio": audio}

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