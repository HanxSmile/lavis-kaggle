import datasets
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
from datasets import load_dataset, Audio, concatenate_datasets
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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


def trim_silence(arr):
    try:
        _max = max(max(arr), -min(arr))
        old_length = len(arr)

        threshold = 30

        for i, e in enumerate(arr):
            if threshold * e > _max:
                break

        for j, e in enumerate(reversed(arr)):
            if threshold * e > _max:
                break

        arr = arr[i:old_length - j]
    except:
        pass
    return arr


class Wav2VecBengaliShrutilipi(torch_Dataset):
    DATASET_NAME = "ucalyptus/shrutilipi_bengali"

    def __init__(self, cache_file, processor, transform=None):
        self.inner_dataset = datasets.load_dataset(self.DATASET_NAME, cache_dir=cache_file)["train"]
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
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        input_secs = input_length / 16_000
        if input_secs <= 1 or input_secs >= 10:
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(remove_special_characters(ann["transcriptions"]))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": str(index),
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
        all_keys = ["input_values", "labels", "attention_mask", "sentences", "ids"]
        result = {k: batch[k] for k in all_keys}
        return result


class Wav2VecBengaliOpenSLR(torch_Dataset):
    def __init__(self, ann_file, data_root, processor, transform=None):
        self.inner_dataset = pd.read_table(ann_file, names=["id", "hash", "sentence"])
        self.processor = processor
        self.media_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset.iloc[index]
        id_ = ann.id
        audio_path = osp.join(self.media_root, id_[:2], id_ + ".flac")
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
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        input_secs = input_length / 16_000
        if input_secs <= 1 or input_secs >= 10:
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor.tokenizer(sentence).input_ids

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
        all_keys = ["input_values", "labels", "attention_mask", "sentences", "ids"]
        result = {k: batch[k] for k in all_keys}
        return result


class Wav2VecBengaliCVBN(torch_Dataset):
    DATASET_NAME = "/mnt/petrelfs/share_data/hanxiao/cvbn"

    def __init__(self, processor, split: str, transform=None):
        split = split.lower()
        assert split in ("train", "validation")
        whole_dataset = datasets.load_from_disk(self.DATASET_NAME)
        self.inner_dataset = concatenate_datasets([whole_dataset["train"], whole_dataset["validation"]])
        self.inner_dataset = self.inner_dataset.filter(lambda x, y: x > y, input_columns=["up_votes", "down_votes"])
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
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        input_secs = input_length / 16_000
        if input_secs <= 1 or input_secs >= 10:
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(remove_special_characters(ann["sentence"]))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": str(index),
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
        all_keys = ["input_values", "labels", "attention_mask", "sentences", "ids"]
        result = {k: batch[k] for k in all_keys}
        return result


class Wav2VecBengaliASR(torch_Dataset):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None):
        split = split.lower()
        split_style = split_style.lower()
        assert split_style in ("k-fold", "default")
        if split_style == "k-fold":
            assert fold_idx is not None and fold_nums is not None and seed is not None
            assert fold_idx in list(range(1, fold_nums + 1))
        assert split in ("train", "valid")

        self.split_style = split_style
        self.fold_idx = fold_idx
        self.fold_nums = fold_nums
        self.seed = seed
        self.split = split

        self.processor = processor
        self.media_root = osp.join(data_root, "train_mp3s")
        self.anno_path = osp.join(data_root, "train.csv")

        self.inner_dataset = self._extract_data()
        self.transform = transform
        self.split = split

    def _extract_data(self):
        annotations = pd.read_csv(self.anno_path)
        if self.split_style == "default":
            data = annotations[annotations["split"] == self.split]
        else:
            Fold = MultilabelStratifiedKFold(n_splits=self.fold_nums, shuffle=True, random_state=self.seed)
            for n, (train_index, val_index) in enumerate(Fold.split(annotations, annotations[["sentence", "split"]])):
                annotations.loc[val_index, 'fold'] = int(n + 1)
            annotations['fold'] = annotations['fold'].astype(int)
            if self.split == "train":
                data = annotations[annotations['fold'] != self.fold_idx].reset_index(drop=True)
            else:
                data = annotations[annotations['fold'] == self.fold_idx].reset_index(drop=True)
        data["audio"] = self.media_root + os.sep + data["id"] + ".mp3"
        return data

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
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        input_secs = input_length / 16_000
        if (input_secs <= 1 or input_secs >= 10) and self.split == "train":
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann.id,
                "input_length": input_length, "audio": audio}

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
        batch["raw_audios"] = [_["audio"] for _ in features]
        all_keys = ["input_values", "labels", "attention_mask", "sentences", "ids", "raw_audios"]
        result = {k: batch[k] for k in all_keys}
        return result


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
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann.file,
                "input_length": input_length, "audio": audio}

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
        batch["raw_audios"] = [_["audio"] for _ in features]
        all_keys = ["input_values", "labels", "attention_mask", "sentences", "ids", "raw_audios"]
        result = {k: batch[k] for k in all_keys}
        return result
