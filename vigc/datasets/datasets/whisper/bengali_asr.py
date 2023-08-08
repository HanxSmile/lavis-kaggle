from torch.utils.data import Dataset
import pandas as pd
import os
import os.path as osp
import datasets
from datasets import Dataset, Audio
import torch
from typing import Dict, List, Union

SPLIT = {
    "train": datasets.Split.TRAIN,
    "valid": datasets.Split.TEST
}


class BengaliASR(Dataset):
    def __init__(self, feature_extractor, tokenizer, processor, data_root, split: str, transform=None):
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
        inner_dataset = Dataset.from_pandas(data, split=SPLIT[split])
        inner_dataset = inner_dataset.cast_column("audio", Audio())
        inner_dataset = inner_dataset.remove_columns(["__index_level_0__", "split"])
        self.transform = transform

        self.inner_dataset = inner_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        audio = ann["audio"]
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        input_features = self.audio_processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        labels = self.text_processor(ann["sentence"]).input_ids
        id_ = ann["id"]
        return {"input_features": input_features, "labels": labels, "sentence": ann["sentence"], "id": id_}

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

        batch["labels"] = labels
        batch["sentences"] = [_["sentence"] for _ in features]
        batch["ids"] = [_["id"] for _ in features]
        return batch
