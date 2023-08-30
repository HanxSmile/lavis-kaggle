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
from datasets import Audio, concatenate_datasets
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

bnorm = Normalizer()

pd.options.mode.chained_assignment = None

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…]'  # 全部符号都去除
# chars_to_ignore_regex = '[\.\-\;\:\"\—\‘\'\‚\“\”\…]'  # 保留 , ? !


def normalize(sentence):
    sentence = re.sub(chars_to_ignore_regex, '', sentence) + " "
    _words = [bnorm(word)['normalized'] for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    return sentence


def trim_silence_(arr):
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


def trim_silence(arr):
    try:
        _max = max(max(arr), -min(arr))
        old_length = len(arr)

        threshold = 30

        for i, e in enumerate(arr):
            if abs(threshold * e) > _max:
                break

        for j, e in enumerate(reversed(arr)):
            if abs(threshold * e) > _max:
                break

        arr = arr[i:old_length - j]
    except:
        pass
    return arr


class Wav2VecBase(torch_Dataset):

    def __init__(self, inner_dataset, processor, transform=None):
        self.inner_dataset = inner_dataset
        self.processor = processor
        self.transform = transform

    def _parse_ann_info(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.inner_dataset)

    def is_valid(self, input_values):
        input_length = len(input_values)
        input_secs = input_length / 16_000
        return input_secs > 1 and input_secs < 10

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        return audio

    def __getitem__(self, index):
        audio, sentence, ann_id = self._parse_ann_info(index)
        audio = self.transform_array(audio)
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        if not self.is_valid(input_values):
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(sentence)
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann_id,
                "input_length": len(input_values), "audio": audio}

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


class Wav2VecBengaliShrutilipi(Wav2VecBase):
    DATASET_NAME = "ucalyptus/shrutilipi_bengali"

    def __init__(self, cache_file, processor, transform=None):
        inner_dataset = datasets.load_dataset(self.DATASET_NAME, cache_dir=cache_file)["train"]
        super().__init__(inner_dataset, processor, transform)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        return ann["audio"], ann["transcriptions"], str(index)


class Wav2VecBengaliOpenSLR(Wav2VecBase):
    def __init__(self, ann_file, data_root, processor, transform=None):
        self.media_root = data_root
        inner_dataset = pd.read_table(ann_file, names=["id", "hash", "sentence"])
        super().__init__(inner_dataset, processor, transform)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        ann_id = ann.id
        audio_path = osp.join(self.media_root, ann_id[:2], ann_id + ".flac")
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        return audio, ann.sentence, ann_id


class Wav2VecBengaliCVBN(Wav2VecBase):
    DATASET_NAME = "/mnt/petrelfs/share_data/hanxiao/cvbn"

    def __init__(self, processor, split: str = "", transform=None):
        whole_dataset = datasets.load_from_disk(self.DATASET_NAME)
        inner_dataset = concatenate_datasets([whole_dataset["train"], whole_dataset["validation"]])
        inner_dataset = inner_dataset.filter(lambda x, y: x > y, input_columns=["up_votes", "down_votes"])
        inner_dataset = inner_dataset.remove_columns(
            ['up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'])
        inner_dataset = inner_dataset.cast_column("audio", Audio(sampling_rate=16_000))

        super().__init__(inner_dataset, processor, transform)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        return ann["audio"], ann["sentence"], str(index)


class Wav2VecBengaliASR(Wav2VecBase):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None, sample_nums=None):
        split = split.lower()
        split_style = split_style.lower()
        assert split_style in ("k-fold", "default")
        if split_style == "k-fold":
            assert fold_idx is not None and fold_nums is not None and seed is not None
            assert fold_idx in list(range(1, fold_nums + 1))
        assert split in ("train", "valid")
        self.split = split
        self.split_style = split_style
        self.fold_idx = fold_idx
        self.fold_nums = fold_nums
        self.seed = seed
        self.split = split
        self.sample_nums = sample_nums

        self.media_root = osp.join(data_root, "train_mp3s")
        self.anno_path = osp.join(data_root, "train.csv")

        inner_dataset = self._extract_data()
        super().__init__(inner_dataset, processor, transform)

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
                if self.sample_nums is None:
                    pass
                data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True).head(self.sample_nums)
        data["audio"] = self.media_root + os.sep + data["id"] + ".mp3"
        return data

    def is_valid(self, input_values):
        if self.split != "train":
            return True
        input_length = len(input_values)
        input_secs = input_length / 16_000
        return input_secs > 1 and input_secs < 10

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = ann.audio
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        return audio, ann.sentence, ann.id


class Wav2VecBengaliASRTest(Wav2VecBase):
    def __init__(self, processor, data_root):
        self.media_root = osp.join(data_root, "examples")
        self.anno_path = osp.join(data_root, "annoated.csv")
        inner_dataset = pd.read_csv(self.anno_path, sep="\t")
        super().__init__(inner_dataset, processor)

    def is_valid(self, input_values):
        return True

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = osp.join(self.media_root, ann.file)
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        return audio, ann.sentence, ann.file
