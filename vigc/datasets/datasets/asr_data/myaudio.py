from torch.utils.data import Dataset as torch_Dataset
import torch
from typing import Dict, List, Union
import numpy as np
import os.path as osp
import librosa
import pandas as pd

MIN_SECS = 1
MAX_SECS = 30
TARGET_SR = 16_000


class MyAudioTest(torch_Dataset):
    def __init__(self, data_root, processor, max_label_length=448):
        self.data_root = data_root
        self.inner_dataset = pd.read_excel(osp.join(data_root, "myaudio_tiny.xlsx"))
        self.processor = processor
        self.transform = None
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.inner_dataset)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = osp.join(self.data_root, ann.audio)
        sentence = ann.text

        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR), TARGET_SR
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }

        return audio, sentence, str(index)

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

        # if not self.is_valid(audio["array"]):
        #     return self[(index + 1) % len(self)]  # filter too long or too short audio

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
